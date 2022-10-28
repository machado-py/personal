import scrapy
#import dateparser
from noticias.items import NoticiasItem

# artigo de referência:============================================================================================================
# https://medium.com/@marlessonsantana/utilizando-o-scrapy-do-python-para-monitoramento-em-sites-de-not%C3%ADcias-web-crawler-ebdf7f1e4966


class AeromagazineSpider(scrapy.Spider):
    name = 'aeromagazine'
    start_urls = ['https://aeromagazine.uol.com.br/artigo/']


    def parse(self, response):
        #//div/h3/a/@href
        
        artigos = response.xpath("//div/h3/a")
        for artigo in artigos:
            link = artigo.xpath("./@href").get()

            yield response.follow(link, self.parse_artigo)


    def parse_artigo(self, response):
        site = self.name
        link = response.url
        titulo = response.xpath("//div/h1/text()").get()
        categoria = 'não tem'
        texto = "".join(response.css("div p ::text").getall())
        autor = response.xpath("//div/h5/text()").get()
        data = response.xpath("//div/h5/p/text()").get()
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
