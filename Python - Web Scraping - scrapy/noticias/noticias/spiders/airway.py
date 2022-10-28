import scrapy
#import dateparser
from noticias.items import NoticiasItem

# artigo de referência:============================================================================================================
# https://medium.com/@marlessonsantana/utilizando-o-scrapy-do-python-para-monitoramento-em-sites-de-not%C3%ADcias-web-crawler-ebdf7f1e4966


class AirwaySpider(scrapy.Spider):
    name = 'airway'
    start_urls = ['https://www.airway.com.br/category/aviacao-comercial/',
                    'https://www.airway.com.br/category/aviacao-militar/',
                    'https://www.airway.com.br/category/aviacao-executiva/']


    def parse(self, response):
        #//div/h2/a/@href
        
        artigos = response.xpath("//div/h2/a")
        for artigo in artigos:
            link = artigo.xpath("./@href").get()

            yield response.follow(link, self.parse_artigo)


    def parse_artigo(self, response):
        site = self.name
        link = response.url
        titulo = response.xpath("//div/h1//span/text()").get()
        categoria = response.xpath('//div[@id="primary"]//ul[@class="post-categories"]/li/a/text()').get()
        texto = "".join(response.css("div p ::text").getall())
        autor = response.xpath("//div[contains(@class, 'author')]/a/text()").get()
        data = response.xpath("//div[contains(@class, 'meta-date')]/text()").get()
        #data = dateparser.parse(data)
        
        noticia = NoticiasItem( link = link,
                                titulo = titulo,
                                categoria = categoria,
                                autor = autor,
                                data = data,
                                texto= texto,
                                site = site
                            )

        yield noticia
