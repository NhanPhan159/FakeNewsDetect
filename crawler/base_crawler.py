from bs4 import BeautifulSoup
import requests
import lxml

class BaseCrawler:
    def __init__(self,link):
        self.request = requests.get(link)
        self.soup = BeautifulSoup(self.request.content,features="lxml")
    def getContentPage(self):
        return self.soup
