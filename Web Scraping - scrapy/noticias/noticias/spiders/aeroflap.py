import scrapy
#import dateparser
from noticias.items import NoticiasItem

# artigo de referência:============================================================================================================
# https://medium.com/@marlessonsantana/utilizando-o-scrapy-do-python-para-monitoramento-em-sites-de-not%C3%ADcias-web-crawler-ebdf7f1e4966


class AeroflapSpider(scrapy.Spider):
    name = 'aeroflap'
    start_urls = ['https://www.aeroflap.com.br/cat/noticias-da-aviacao/aeronaves-news/',
                'https://www.aeroflap.com.br/cat/noticias-da-aviacao/companhias-aereas-news/',
                'https://www.aeroflap.com.br/cat/noticias-da-aviacao/empresas-news/']


    def parse(self, response):
        #//div[@class='td-ss-main-content']//div[@class='item-details']/h3/a/@href
        
        artigos = response.xpath("//div[@class='td-ss-main-content']//div[@class='item-details']")
        for artigo in artigos:
            link = artigo.xpath("./h3/a/@href").get()

            yield response.follow(link, self.parse_artigo)


    def parse_artigo(self, response):
        site = self.name
        link = response.url
        titulo = response.xpath("//div[@class='td-post-header']/header/h1/text()").get()
        categoria = response.xpath("//div[@class='td-post-header']//span[3]//a/text()").get()
        texto = "".join(response.css("div p ::text").getall())

        data = response.xpath("//div[@class='td-post-header']/header//span/time/text()").get()
        #data = dateparser.parse(data)
        
        autor = response.xpath("//em/text()").get()
        if autor is None:
            autor = response.xpath("//div[contains(@class,'author')]/span/a/text()").get()      

        noticia = NoticiasItem( link = link,
                                titulo = titulo,
                                categoria = categoria,
                                autor = autor,
                                data = data,
                                texto= texto,
                                site = site
                            )

        yield noticia
