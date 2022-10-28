# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class NoticiasItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    link = scrapy.Field()
    titulo = scrapy.Field()
    categoria = scrapy.Field()
    autor = scrapy.Field()
    data = scrapy.Field()
    texto = scrapy.Field()
    site = scrapy.Field()
    
