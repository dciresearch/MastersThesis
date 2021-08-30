from datetime import datetime
from datetime import timedelta

import scrapy
from .news import NewsSpider
from .news import NewsSpiderConfig
from scrapy.linkextractors import LinkExtractor


class KommersantSpider(NewsSpider):
    name = "kommersant"

    base_url = "https://www.kommersant.ru"
    link_tmpl = "https://www.kommersant.ru/archive/news/day/{}"
    # Start with the current date

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        link_tmpl = "https://www.kommersant.ru/archive/news/day/{}"
        self.page_dt = self.start_date
        self.start_urls = [link_tmpl.format(self.page_dt.strftime("%Y-%m-%d"))]

    # Ignore "robots.txt" for this spider only
    custom_settings = {"ROBOTSTXT_OBEY": "False"}

    config = NewsSpiderConfig(
        title_path='//h1[contains(@class, "article_name")]//text()',
        date_path='//time[contains(@class, "title")]/@datetime',
        date_format="%Y-%m-%dT%H:%M:%S%z",  # 2019-03-09T12:03:10+03:00
        text_path='//p[@class="b-article__text"]',
        tags_path='//meta[contains(@name, "keywords")]/@content'
    )
    news_le = LinkExtractor(
        restrict_xpaths='//li[contains(@class,"result__item")]')

    def parse(self, response):
        # Parse most recent news
        for i in self.news_le.extract_links(response):
            yield scrapy.Request(url=i.url,
                                 callback=self.parse_document,
                                 meta={"page_dt": self.page_dt})

        # If it's not the end of the page, request more news from archive by calling recursive "parse_page" function
        more_link = response.xpath(
            '//button[contains(@class, "lazyload-button")]/@data-lazyload-url'
        ).extract()
        if more_link:
            yield scrapy.Request(
                url="{}{}".format(self.base_url, more_link[0]),
                callback=self.parse_page,
                meta={"page_dt": self.page_dt},
            )

        # Requesting the next page if we need to
        self.page_dt -= timedelta(days=1)
        if self.start_date >= self.page_dt > self.until_date:
            link_url = self.link_tmpl.format(self.page_dt.strftime("%Y-%m-%d"))

            yield scrapy.Request(
                url=link_url,
                priority=100,
                callback=self.parse,
                meta={
                    "page_depth": response.meta.get("page_depth", 1) + 1,
                    "page_dt": self.page_dt,
                },
            )

    def parse_page(self, response):
        # Parse all articles on page
        for i in self.news_le.extract_links(response):
            yield scrapy.Request(url=i.url, callback=self.parse_document)

        # Take a link from "more" button
        more_link = response.xpath(
            '//button[contains(@class, "lazyload-button")]/@data-lazyload-url'
        ).extract()
        if more_link:
            yield scrapy.Request(
                url="{}{}".format(self.base_url, more_link[0]),
                callback=self.parse_page,
                meta={
                    "page_depth": response.meta.get("page_depth", 1),
                    "page_dt": response.meta["page_dt"],
                },
            )

    def parse_document(self, response):
        for res in super().parse_document(response):
            # If it's a gallery (no text) or special project then don't return anything (have another html layout)
            if "text" not in res or "title" not in res:
                break
            res["tags"] = [
                tags.strip(",").replace(",", ", ") for tags in res["tags"]
            ]
            art_body = []
            art_links = []
            for t in res["text"]:
                par = t.xpath('string(.)').get().replace("\n", "")
                links = t.xpath('.//a/@href').getall()
                art_body.append(par)
                art_links += links
            res["text"]=art_body
            res["links"]=art_links
            # Remove ":" in timezone
            pub_dt = res["date"][0]
            res["date"] = [pub_dt[:-3] + pub_dt[-3:].replace(":", "")]

            yield res