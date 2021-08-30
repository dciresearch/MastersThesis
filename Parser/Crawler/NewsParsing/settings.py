# Scrapy settings for NewsParsing project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://docs.scrapy.org/en/latest/topics/settings.html
#     https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://docs.scrapy.org/en/latest/topics/spider-middleware.html


from shutil import which

SELENIUM_DRIVER_NAME = 'chrome'

SELENIUM_DRIVER_EXECUTABLE_PATH = "C:\\Users\\Orbis\\Downloads\\chromedriver_win32(4)\\chromedriver.exe"

SELENIUM_DRIVER_ARGUMENTS=['-headless']

BOT_NAME = 'NewsParsing'

SPIDER_MODULES = ['NewsParsing.spiders']
NEWSPIDER_MODULE = 'NewsParsing.spiders'


# Crawl responsibly by identifying yourself (and your website) on the user-agent
#USER_AGENT = 'NewsParsing (+http://www.yourdomain.com)'

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

USER_AGENT = "newsbot"

ROBOTSTXT_OBEY = True

# Configure maximum concurrent requests performed by Scrapy (default: 16)
CONCURRENT_REQUESTS = 512
CONCURRENT_REQUESTS_PER_DOMAIN = 128
CONCURRENT_REQUESTS_PER_IP = 128

AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1
AUTOTHROTTLE_MAX_DELAY = 0.5
AUTOTHROTTLE_TARGET_CONCURRENCY = 128

DOWNLOAD_DELAY = 0.1
RANDOMIZE_DOWNLOAD_DELAY = True
# The download delay setting will honor only one of:
# CONCURRENT_REQUESTS_PER_DOMAIN = 16
# CONCURRENT_REQUESTS_PER_IP = 16

RETRY_ENABLED = True
RETRY_TIMES = 3
RETRY_HTTP_CODES = [500, 502, 503, 504, 400, 401, 403, 404, 405, 406, 407, 408, 409, 410, 429]


# Disable cookies (enabled by default)
COOKIES_ENABLED = False

ITEM_PIPELINES = {
    "NewsParsing.pipelines.NewsparsingPipeline": 300,
}

DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
    'scrapy.spidermiddlewares.referer.RefererMiddleware': 80,
    'scrapy.downloadermiddlewares.retry.RetryMiddleware': 90,
    'scrapy_fake_useragent.middleware.RandomUserAgentMiddleware': 120,
    'scrapy.downloadermiddlewares.cookies.CookiesMiddleware': 130,
    'scrapy.downloadermiddlewares.httpcompression.HttpCompressionMiddleware': 810,
    'scrapy.downloadermiddlewares.redirect.RedirectMiddleware': 900,
    #'scrapy_selenium.SeleniumMiddleware': 800
}