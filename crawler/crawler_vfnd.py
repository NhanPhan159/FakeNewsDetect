from base_crawler_for_one import BaseCrawlerForOne
import urllib.request
import json
class CrawlerVFND(BaseCrawlerForOne):
    def __init__(self, url, nguon, label=1, loai=None):
        super(CrawlerVFND,self).__init__(url, nguon, label, loai)
    
    def getLink(self):
        news_feeds = self.content_page.find("div", class_="js-details-container Details").find_all('div', class_="flex-auto min-width-0 col-md-2 mr-3")
        for news_feed in news_feeds:
            href = news_feed.find('a')
            if ("README.md" in href ) or ("VFND_Ac_Real_1.json" in href):
                continue
            link = "https://raw.githubusercontent.com" + href.get('href') 
            link = link.replace("/blob","")
            self.links.append(link)

    def getNewsOnPage(self,link):
        # lay data dang json
        page = urllib.request.urlopen(link)
        data = page.read()
        encoding = page.info().get_content_charset('utf-8')
        JSON_object = json.loads(data.decode(encoding))
        ## lay tieu de
        news_title = JSON_object['title']
        ## lay thoi gian
        news_time = JSON_object['date_publish']
        ## lay noi dung
        try:
            news_body = JSON_object['maintext']
        except:
            news_body = JSON_object['text']
        ## lay nguon
        news_page = JSON_object['source_domain']
        ## lay url
        news_url = JSON_object['url']
        return news_title,news_time,news_body, news_page, news_url
    
    def getFullInformationNews(self,link): # trung
        label = self.label
        news_title,news_time,news_body,news_page,news_url = self.getNewsOnPage(link)

        self.data['nhan'].append(label)
        self.data['loai'].append(self.loai)
        self.data['tieu_de'].append(news_title)
        self.data['tg_dang_tin'].append(news_time)
        self.data['noi_dung'].append(news_body)
        self.data['nguon'].append(news_page)
        self.data['link'].append(news_url)
        print("Crawl tin tuc: ",news_title)

# if __name__ == '__main__':
#     crawler = CrawlerVFND(url="https://github.com/VFND/VFND-vietnamese-fake-news-datasets/tree/master/Fake_Real_Dataset/Fake/Article_Contents",
#                           nguon="",
#                           label=0,
#                           loai="xa_luan")
#     crawler.collectDataToFrame()