from base_crawler_for_one import BaseCrawlerForOne
from base_crawler import BaseCrawler
import time
import pandas as pd

class CrawlerNlb(BaseCrawlerForOne):
    def __init__(self,url,nguon,label=1,loai=None) :
        super(CrawlerNlb, self).__init__(url,nguon,label,loai)
        self.time_in_link = []

    def getLink(self):
        content = self.content_page.find("div", class_="blog-posts hfeed")
        new_feeds = content.find_all('h2', class_="post-title")
        date_times = content.find_all('abbr', class_="published")
        for new_feed in new_feeds:
            href = new_feed.find('a')
            link = href.get('href')
            self.links.append(link)
        for date_time in date_times:
            self.time_in_link.append(date_time.text)

    def getNewsOnPage(self,link):
        content_page = BaseCrawler(link).getContentPage()
        ## lay tieu de
        news_title = content_page.find('h1',class_='post-title entry-title').text
        ## lay noi dung
        news_body_element = content_page.find('div',class_="post-body entry-content")
        news_body = news_body_element.text
        return news_title,news_body
    
    def getFullInformationNews(self,link,ids=False):
        if ids==False:
            label = self.label
            news_title,news_body = self.getNewsOnPage(link)
            if self.loai==None:
                self.loai = self.url.replace(self.nguon,"").replace("/",'')

            self.data['nhan'].append(label)
            self.data['loai'].append(self.loai)
            self.data['tieu_de'].append(news_title)
            self.data['noi_dung'].append(news_body)
            self.data['nguon'].append(self.nguon)
            self.data['link'].append(link)
            print("Crawl tin tuc: ",news_title)

    def collectDataToFrame(self):
        self.getLink()
        for link in self.links:
            self.getFullInformationNews(link=link)
            time.sleep(1)
        self.data['tg_dang_tin'] = self.time_in_link
        self.format_pandas = pd.DataFrame(self.data)
        print("crawl hoàn thành")
        
if __name__ == '__main__':
    crawler = CrawlerNlb("https://danlambaovn.blogspot.com/#",nguon="https://danlambaovn.blogspot.com/#")
    crawler.collectDataToFrame()

        