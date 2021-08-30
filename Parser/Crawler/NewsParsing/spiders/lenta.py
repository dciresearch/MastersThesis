import datetime
import re

from .news import NewsSpider
from .news import NewsSpiderConfig


class LentaSpider(NewsSpider):
    name = "lenta"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dt=self.start_date
        self.link_tmpl = "https://lenta.ru/news/{}"
        self.start_urls = [self.link_tmpl.format(dt.strftime("%Y/%m/%d"))]
    config = NewsSpiderConfig(
        title_path='//h1[contains(@itemprop, "headline")]/text()',
        date_path='//meta[contains(@itemprop, "datePublished")]/@content',
        date_format="%Y-%m-%dT%H:%M:%S%z",
        text_path=
        '//div[contains(@itemprop, "articleBody")]/p',
        tags_path='_',
    )

    def parse(self, response):
        page_date = self.start_date
        while self.start_date >= page_date > self.until_date:
            url = self.link_tmpl.format(page_date.strftime("%Y/%m/%d"))
            yield response.follow(url, self.parse_page)

            page_date -= datetime.timedelta(days=1)

    def parse_page(self, response):
        url = response.url
        for document_href in response.xpath(
                '//div[contains(@class, "item news")]/div[contains(@class, "titles")]/h3/a/@href').extract():
            yield response.follow(document_href, self.parse_document)

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