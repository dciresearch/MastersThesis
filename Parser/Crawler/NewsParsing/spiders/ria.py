import datetime
import re

import scrapy
from .news import NewsSpider
from .news import NewsSpiderConfig
from scrapy_selenium import SeleniumRequest


class RiaSpider(NewsSpider):
    name = "ria"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dt=self.start_date
        self.link_tmpl = "https://ria.ru/services/{0}/more.html?&date={0}T"
        self.start_urls = [self.link_tmpl.format(self.dt.strftime("%Y%m%d"))+"235959"]
    config = NewsSpiderConfig(
        title_path='//h1[contains(@class, "article__title")]/text()',
        date_path='//meta[contains(@property, "article:published_time")]/@content',
        date_format="%Y%m%dT%H%M",
        text_path=
        '//div[contains(@class, "article__text")]',
        tags_path='//meta[contains(@name, "analytics:tags")]/@content',
    )

    custom_settings = {"DOWNLOAD_DELAY": 0.2}


    def parse(self, response):
        if self.start_date >= self.dt > self.until_date:
            time=response.xpath('//div[contains(@class, "list-item__date")]/text()').extract()
            for document_href in response.xpath(
                    '//a[contains(@class, "list-item__title")]/@href'
                    ).extract():
                yield response.follow(document_href, self.parse_document)

            if len(set(time))>1:
                url=self.link_tmpl.format(self.dt.strftime("%Y%m%d"))+time[-1].split(',')[-1].replace(":","").strip()+"00"
                yield scrapy.Request(url=url, priority=100, callback=self.parse)
            else:
                self.dt -= datetime.timedelta(days=1)
                url = self.link_tmpl.format(self.dt.strftime("%Y%m%d"))+"235959"
                yield scrapy.Request(url=url,priority=1000, callback=self.parse)

    def parse_document(self, response):
        for res in super().parse_document(response):

            art_body = []
            art_links = []
            for t in res["text"]:
                par = t.xpath('string(.)').get().replace("\n", "")
                links = t.xpath('.//a/@href').getall()
                art_body.append(par)
                art_links += links
            res["text"] = art_body
            res["links"] = art_links
            res["tags"]=res["tags"][0].split(',')
            # Remove ":" in timezone
            pub_dt = res["date"][0]
            res["date"] = [pub_dt[:-3] + pub_dt[-3:].replace(":", "")]

            yield res