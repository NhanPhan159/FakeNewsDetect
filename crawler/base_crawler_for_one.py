from base_crawler import BaseCrawler
import time
import pandas as pd
class BaseCrawlerForOne:
    def __init__(self,url,nguon,label=1,loai=None) :
        self.content_page = BaseCrawler(url).getContentPage()
        self.data = {
            "loai": [],
            "nguon": [],
            "link": [],
            "tg_dang_tin":[],
            "tieu_de":[],
            "noi_dung":[],
            "nhan":[]
        }
        self.nguon = nguon
        self.url = url
        self.label = label
        self.format_pandas = None
        self.links = []
        self.loai = loai
    
    def getLink(self):
        pass
    def getNewsOnPage(self):
        pass

    def getFullInformationNews(self,link): # trung
        label = self.label
        news_title,news_time,news_body = self.getNewsOnPage(link)
        if self.loai == None :
            self.loai = self.url.replace(self.nguon,"").replace("/",'')

        self.data['nhan'].append(label)
        self.data['loai'].append(self.loai)
        self.data['tieu_de'].append(news_title)
        self.data['tg_dang_tin'].append(news_time)
        self.data['noi_dung'].append(news_body)
        self.data['nguon'].append(self.nguon)
        self.data['link'].append(link)
        print("Crawl tin tuc: ",news_title)
    def collectDataToFrame(self):
        
        self.getLink()
        for link in self.links:
            self.getFullInformationNews(link=link)
            time.sleep(1)
        self.format_pandas = pd.DataFrame(self.data)
    def getDataAfterCrawl(self):
        return self.format_pandas