import scrapy
#import dateparser
from noticias.items import NoticiasItem

# artigo de referência:============================================================================================================
# https://medium.com/@marlessonsantana/utilizando-o-scrapy-do-python-para-monitoramento-em-sites-de-not%C3%ADcias-web-crawler-ebdf7f1e4966


class AirwaySpider(scrapy.Spider):
    name = 'aviacaobrasil'
    start_urls = ['https://aviacaobrasil.com.br/category/noticias/fabricantes/',
                    'https://aviacaobrasil.com.br/category/noticias/mro/',
                    'https://aviacaobrasil.com.br/category/noticias/financas/',
                    'https://aviacaobrasil.com.br/category/noticias/tecnologia-ti/', 
                    'https://aviacaobrasil.com.br/category/noticias/aviacao-executiva/',
                    'https://aviacaobrasil.com.br/category/noticias/aliancas/']


    def parse(self, response):
        #//h3/a/@href
        
        artigos = response.xpath("//h3/a")
        for artigo in artigos:
            link = artigo.xpath("./@href").get()

            yield response.follow(link, self.parse_artigo)


    def parse_artigo(self, response):
        site = self.name
        link = response.url
        titulo = response.xpath("//header/h1/text()").get()
        categoria = 'não tem'
        texto = "".join(response.css("div p ::text").getall())
        autor = response.xpath("//header/div/div/a/text()").get()
        data = response.xpath("//header/div/span/time/text()").get()
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
