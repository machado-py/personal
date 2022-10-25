# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
#from itemadapter import ItemAdapter

# TESTE1
# import json

# class NoticiasPipeline:
#     def open_spider(self, spider):
#         self.file = open('noticias.txt', 'w', encoding='utf-8')
    
#     def close_spider(self, spider):
#         self.file.close
        
#     def process_item(self, item, spider):
#         line = json.dumps(dict(item)) + '\n'
#         self.file.write(line)
#         return item



# TESTE2
from itemadapter import ItemAdapter
from scrapy.exporters import JsonItemExporter

class NoticiasPipeline:
    """Distribute items across multiple files according to their choosed field"""

    def open_spider(self, spider):
        self.item_to_exporter = {}

    def close_spider(self, spider):
        for exporter, saved_file in self.item_to_exporter.values():
            exporter.finish_exporting()
            saved_file.close()

    def _exporter_for_item(self, item):
        adapter = ItemAdapter(item)
        item_selected = adapter['site']
        if item_selected not in self.item_to_exporter:
            saved_file = open(f'{item_selected}.json', 'wb', )
            exporter = JsonItemExporter(saved_file)
            exporter.start_exporting()
            self.item_to_exporter[item_selected] = (exporter, saved_file)
        return self.item_to_exporter[item_selected][0]

    def process_item(self, item, spider):
        exporter = self._exporter_for_item(item)
        exporter.export_item(item)
        return item