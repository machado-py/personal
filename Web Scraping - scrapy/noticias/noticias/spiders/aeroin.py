import scrapy
# import dateparser
from noticias.items import NoticiasItem

# artigo de referência:============================================================================================================
# https://medium.com/@marlessonsantana/utilizando-o-scrapy-do-python-para-monitoramento-em-sites-de-not%C3%ADcias-web-crawler-ebdf7f1e4966


class AeroinSpider(scrapy.Spider):
    name = 'aeroin'
    urls = ['https://www.aeroin.net/category/empresas-aereas/page/',
            'https://www.aeroin.net/category/industria-2/page/',
            'https://www.aeroin.net/category/aeroportos-e-servicos/page/' ]

    start_urls = [url + str(pagina) +'/' for url in urls for pagina in range(1,4)]


    def parse(self, response):
        #//div[@id='tdi_60']//div[@class='td-module-meta-info']/h3/a/@href
        
        artigos = response.xpath("//div[@id='tdi_60']//div[@class='td-module-meta-info']")
        for artigo in artigos:
            link = artigo.xpath("./h3/a/@href").extract_first()

            yield response.follow(link, self.parse_artigo)


    def parse_artigo(self, response):
        site = self.name
        link = response.url
        titulo = response.xpath("//div/h1/text()").extract_first()
        categoria = response.xpath("//div/a[@class='tdb-entry-category']/text()").extract_first()
        autor = response.xpath("//div/a[@class='tdb-author-name']/text()").extract_first()
        data = response.xpath("//div/time/text()").extract_first()
        texto = "".join(response.css("div p ::text").extract())

        if titulo is None:
            titulo = response.xpath("//div/header/h1/text()").extract_first()
        
        if categoria is None:
            categoria = response.xpath("//div[@class='entry-crumbs']/span[2]/a/text()").extract_first()
        
        if autor is None:
            autor = response.xpath("//div[@class='td-module-meta-info']//a/text()").extract_first()
        
        if data is None:
            data = response.xpath("//div[@class='td-module-meta-info']/span/time/text()").extract_first()
        
        if texto is None:
            texto = response.css("div.td-post-content p ::text").extract()
        
        # data = dateparser.parse(data)

        noticia = NoticiasItem( link = link,
                                titulo = titulo,
                                categoria = categoria,
                                autor = autor,
                                data = data,
                                texto= texto,
                                site = site
                            )

        yield noticia

