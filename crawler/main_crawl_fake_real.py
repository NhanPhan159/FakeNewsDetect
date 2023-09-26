from crawler_nd import CrawlerNd
from crawler_vfnd import CrawlerVFND
from crawler_nlb import CrawlerNlb
from data_controller import DataController
import pandas as pd 

if __name__ == '__main__':
    ## khoi tao cac bien gia tri
    url = "https://github.com/VFND/VFND-vietnamese-fake-news-datasets/tree/master/Dataset/Fake/Article_Contents"
    nguon = ""
    label = 1 ## 1 là fake, 0 là real
    loai = "xa_luan"
    path = "../dataset/csv/fake_news_viet.csv"

    ## crawl dữ liệu, đổi lớp tuỳ theo trang báo crawl
    crawler = CrawlerVFND(url=url,
                          nguon=nguon,
                          label=label,
                          loai=loai)
    crawler.collectDataToFrame()

    # xét file csv có tồn tại
    if DataController.checkFileIsExist(path):
        data_controller = DataController(path=path)
        print("File đã tồn tại, tiến hành lưu dữ liệu crawl")
        data_controller.readDataFromPath()
        data_controller.appendData(crawler.getDataAfterCrawl())
    else: # không tồn tại file truoc do
        data_controller = DataController(path=path)
        data_controller.setData(crawler.getDataAfterCrawl())
    data_controller.removeSpecialRow()
    data_controller.saveData()

    #thống kê
    data_controller.staticsData()
