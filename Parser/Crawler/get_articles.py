from scrapy.utils.project import get_project_settings
from scrapy.crawler import CrawlerProcess
from datetime import datetime, timedelta
from multiprocessing.context import Process

start="09.11.2020"
end="01.01.2016"
end_date=datetime.strptime(end,"%d.%m.%Y").date()
cur=datetime.strptime(start,"%d.%m.%Y").date()
spider_names=["ria"]
setting = get_project_settings()

def crawl(s,f):
    crawler = CrawlerProcess(setting)
    for spider_name in spider_names:
        print ("Running spider %s for date %s" % (spider_name, s.strftime("%d-%m-%Y")))
        crawler.crawl(spider_name, start_date=s.strftime("%d.%m.%Y"), until_date=f.strftime("%d.%m.%Y"))
    crawler.start()

if __name__ == '__main__':
    while cur > end_date:
        dates=[(cur - timedelta(days=7*i)) for i in range(0,2)]
        cur=dates[1]
        s=dates[0]
        f=dates[1]
        p = Process(target=crawl, args=(s,f))
        p.start()
        p.join()