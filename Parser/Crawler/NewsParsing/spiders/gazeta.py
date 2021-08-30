import datetime
import re

import scrapy
from .news import NewsSpider
from .news import NewsSpiderConfig


class GazetaSpider(NewsSpider):
    name = "gazeta"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_url="https://www.gazeta.ru"
        self.dt=datetime.datetime.combine(self.start_date, datetime.datetime.max.time())
        self.link_tmpl = "https://www.gazeta.ru/news/?p=page&d={}"
        self.start_urls = [self.link_tmpl.format(self.dt.strftime("%d.%m.%Y_%H:%M"))]
    config = NewsSpiderConfig(
        title_path='//h1[contains(@itemprop, "headline")]/text()',
        date_path='//time[contains(@itemprop, "datePublished")]/@datetime',
        date_format="%Y-%m-%dT%H:%M:%S%z",
        text_path=
        '//div[contains(@itemprop, "articleBody")]/p',
        tags_path='_',
    )

    def parse(self, response):
        if self.start_date >= self.dt.date() > self.until_date:
            art_times=response.xpath('//article[contains(@class, "news_main")]/meta[contains(@itemprop,"dateModified")]/@content').extract()
            min_dt=min([datetime.datetime.strptime("".join(i.rsplit(":",1)), self.config.date_format) for i in art_times])

            for document in response.xpath(
                    '//article[contains(@class, "news_main")]/div[contains(@class, "news_body")]/div'
                    ):
                date=document.xpath(".//h3/text()").extract()[0]
                date = datetime.datetime.strptime(date, "%d.%m.%Y").date()
                document_href=document.xpath(".//div/h1/a/@href").extract()[0]
                if date < self.dt.date():
                    break
                yield response.follow(self.base_url + document_href, self.parse_document)

            if min_dt.date() < self.dt.date():
                self.dt -= datetime.timedelta(days=1)
                date = self.dt
            else:
                date=min_dt
            url = self.link_tmpl.format(date.strftime("%d.%m.%Y_%H:%M"))
            yield scrapy.Request(url=url,priority=100, callback=self.parse)


    def parse_document(self, response):
        for res in super().parse_document(response):

            art_body = []
            art_links = []
            for t in res["text"]:
                par = t.xpath('string(.)').get().replace("\n", "")
                if "НОВОСТИ ПО" in par:
                    break
                links = t.xpath('.//a/@href').getall()
                art_body.append(par)
                art_links += links
            res["text"] = art_body
            res["links"] = art_links

            # Remove ":" in timezone
            pub_dt = res["date"][0]
            res["date"] = [pub_dt[:-3] + pub_dt[-3:].replace(":", "")]

            yield res