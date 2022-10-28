pagina=1
urls = ['https://www.aeroin.net/category/empresas-aereas/page/',
        'https://www.aeroin.net/category/industria-2/page/',
        'https://www.aeroin.net/category/aeroportos-e-servicos/page/' ]

start_urls = [url + str(pagina) +'/' for url in urls for pagina in range(1,3)]

print(start_urls)
