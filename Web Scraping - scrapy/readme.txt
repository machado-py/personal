Treinando biblioteca scrapy.
O objetivo é extrair informações dos links da primeira página de vários sites de notícias.
Atualmente são 5 spiders de sites diferentes.

O arquivo "runallspiders.py" , na pasta noticias/noticias, executa todos os spiders em sequência.
O arquivo "pipelines.py" foi modificado para criar arquivos json da extração de cada site.
O arquivo "items.py" possui os campos extraídos.
O arquivo "settings.py" tem 3 pequenas modificações: ROBOTSTXT_OBEY, ITEM_PIPELINES, AUTOTHROTTLE_ENABLED
