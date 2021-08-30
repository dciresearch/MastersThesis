from urllib.parse import urlsplit
from datetime import datetime, timedelta

import scrapy
from scrapy.loader import ItemLoader

from .. items import Document

class NewsSpiderConfig:
    def __init__(
            self,
            title_path,
            text_path,
            date_path,
            date_format,
            tags_path,
    ):
        self.title_path = title_path
        self.date_path = date_path
        self.date_format = date_format
        self.text_path = text_path
        self.tags_path = tags_path


class NewsSpider(scrapy.Spider):
    def __init__(self, *args, **kwargs):
        assert self.config
        assert self.config.title_path
        assert self.config.date_path
        assert self.config.date_format
        assert self.config.text_path
        assert self.config.tags_path

        # starting date for articles
        if "start_date" in kwargs:
            kwargs["start_date"] = datetime.strptime(kwargs["start_date"],
                                                     "%d.%m.%Y").date()
        else:
            kwargs["start_date"] = datetime.now().date()

        # till date for articles
        if "until_date" in kwargs:
            kwargs["until_date"] = datetime.strptime(kwargs["until_date"],
                                                     "%d.%m.%Y").date()
        else:
            # If there's no 'until_date' param, get articles for starting day
            kwargs["until_date"] = (kwargs["start_date"] - timedelta(days=1))


        super().__init__(*args, **kwargs)

    def parse_document(self, response):
        url = response.url
        base_edition = urlsplit(self.start_urls[0])[1]
        edition = urlsplit(url)[1]
        item=Document()

        item["url"]=url
        item["links"]=[]
        item["text"] = response.xpath(self.config.text_path)
        item["edition"]="-" if edition == base_edition else edition
        item["title"]=response.xpath(self.config.title_path).getall()
        item["date"]=response.xpath(self.config.date_path).getall()
        item["tags"]=response.xpath(self.config.tags_path).getall()

        yield item

    def process_title(self, title):
        return title

    def process_text(self, paragraphs):
        text = "\\n".join([p.strip() for p in paragraphs if p.strip()])
        text = text.replace("\xa0", " ").replace(" . ", ". ").strip()
        return text