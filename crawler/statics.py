from data_controller import DataController

path = "../dataset/csv/real_news.csv"
data_controller = DataController(path=path)
data_controller.readDataFromPath()
data_controller.staticsData()