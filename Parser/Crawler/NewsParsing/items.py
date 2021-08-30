# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class Document(scrapy.Item):
    # define the fields for your item here like:
    url = scrapy.Field()
    title = scrapy.Field()
    text = scrapy.Field()
    links=scrapy.Field()
    edition=scrapy.Field()
    date = scrapy.Field()
    tags = scrapy.Field()
    pass
