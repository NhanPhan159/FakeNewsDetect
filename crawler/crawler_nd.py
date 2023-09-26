from base_crawler_for_one import BaseCrawlerForOne
from base_crawler import BaseCrawler

class CrawlerNd(BaseCrawlerForOne):
    def __init__(self,url,nguon,label=1,loai=None) :
        super(CrawlerNd, self).__init__(url,nguon,label,loai)

    def getLink(self):
        new_feeds = self.content_page.find('div', class_='box-content content-list').find_all('h3', class_='story__heading')
        for new_feed in new_feeds:
            href = new_feed.find('a')
            link = href.get('href')
            self.links.append(link)

    def getNewsOnPage(self,link):
        content_page = BaseCrawler(link).getContentPage()
        ## lay tieu de
        news_title = content_page.find('h1',class_='article__title cms-title').text
        ## lay thoi gian
        news_time = content_page.find('time',class_='time').text
        ## lay noi dung
        news_body_element = content_page.find('div',class_="article__body cms-body").find_all('p')
        news_body = ""
        for text in news_body_element:
            news_body = news_body + " " + text.text
        return news_title,news_time,news_body
        
# if __name__ == '__main__':
#     crawler = CrawlerNd("https://nhandan.vn/xa-luan/",nguon="https://nhandan.vn")
#     crawler.collectDataToFrame()

        