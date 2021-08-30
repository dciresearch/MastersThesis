import datetime
import re

import scrapy
from .news import NewsSpider
from .news import NewsSpiderConfig
from scrapy_selenium import SeleniumRequest


class RainSpider(NewsSpider):
    name = "rain"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dt=self.start_date
        self.base_url='https://tvrain.ru'
        self.link_tmpl = "https://tvrain.ru/archive/?search_teleshow_cat=&search_year={}&search_month={}&search_day={}&query=&tab=2"
        self.start_urls = [self.link_tmpl.format(self.dt.year,self.dt.month, self.dt.day)]
    config = NewsSpiderConfig(
        title_path='//div[contains(@class, "head__title")]/h1/text()',
        date_path='//meta[contains(@property, "published_time")]/@content',
        date_format="%Y-%m-%dT%H:%M:%S%z",
        text_path=
        '//div[contains(@class, "document-lead") or contains(@class, "article-full__text")]/p',
        tags_path='_',
    )

    # Ignore "robots.txt" for this spider only
    custom_settings = {"ROBOTSTXT_OBEY": "False"}


    def parse(self, response):
        dt=self.dt
        if response.meta.get('parsing_page', False):
           dt=response.meta['dt']
        if self.start_date >= dt > self.until_date:
            pages=[]
            if not response.meta.get('parsing_page', False):
                pages=response.xpath('//div[contains(@class, "pagination")]/a/@href').extract()

            for p in pages:
                expanded=self.base_url+p
                yield scrapy.Request(url=expanded, priority=1000, callback=self.parse, meta={'parsing_page': True, "dt": self.dt})

            for document_href in response.xpath(
                    '//a[contains(@class, "chrono_list__item__info__name--nocursor")]/@href'
                    ).extract():
                yield response.follow(self.base_url + document_href, self.parse_document)

            if not response.meta.get('parsing_page', False):
                self.dt -= datetime.timedelta(days=1)
                url = self.link_tmpl.format(self.dt.year,self.dt.month, self.dt.day)
                yield scrapy.Request(url=url,priority=100, callback=self.parse)

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

            # Remove ":" in timezone
            pub_dt = res["date"][0]
            res["date"] = [pub_dt[:-3] + pub_dt[-3:].replace(":", "")]

            yield res