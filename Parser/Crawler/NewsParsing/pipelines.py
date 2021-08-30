# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html
import csv
import datetime
import os

# useful for handling different item types with a single interface
from itemadapter import ItemAdapter


class NewsparsingPipeline:
    def open_spider(self, spider):
        if not os.path.exists("./{}".format(spider.name)):
            os.makedirs("./{}".format(spider.name))
        self.file = open("./{}/{}.csv".format(spider.name, spider.start_date), 'w', encoding='utf8')
        self.fields = ["date", "url","tags", "links", "edition", "title", "text"]
        self.writer = csv.writer(self.file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        self.file.write(','.join(self.fields) + '\n')

    def close_spider(self, spider):
        self.file.close()

    def process_item(self, item, spider):
        dt = datetime.datetime.strptime(item["date"][0], spider.config.date_format)
        item["title"] = spider.process_title(item["title"][0])
        item["text"] = spider.process_text(item["text"])

        item["edition"] = item["edition"][0]
        item["url"] = item["url"]
        item["date"] = dt.strftime("%Y-%m-%d %H:%M:%S")
        item["tags"] = ", ".join(item.get("tags", "-"))
        item["links"]= " ".join(item.get("links", "-"))

        if dt.date() < spider.until_date:
            return item
        line = (item["date"], item["url"], item["tags"], item["links"],
                item["edition"], item["title"], item["text"])
        self.writer.writerow(line)
        return item
