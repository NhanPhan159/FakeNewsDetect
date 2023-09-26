import pandas as pd
import os
import json
import random
class DataController:
    def __init__(self,path=""):
        self.path = path
        self.data_from_path = pd.DataFrame() 
    
    @staticmethod
    def checkFileIsExist(path):
        return os.path.exists(path)
            
    def readDataFromPath(self):
        self.data_from_path = pd.read_csv(self.path)

    def checkIfLinkDuplicate(self, news_link):
        return (news_link in self.data_from_path['link'].values)
    
    def appendData(self,pandas_frame):        
        index_arr = []
        for index in pandas_frame.index:
            if self.checkIfLinkDuplicate(pandas_frame['link'][index]):
                index_arr.append(index)
        pandas_frame_remove_duplicate = pandas_frame.drop(index=index_arr)
        self.data_from_path = pd.concat([self.data_from_path, pandas_frame_remove_duplicate], ignore_index=True)
        print("Data appended !!!")
        
    def setData(self,data_frame):
        self.data_from_path = data_frame

    def saveData(self):
        self.data_from_path.to_csv(self.path, index=False)
        print("Save data !!!")

    def staticsData(self):
        print("Thống kê số lượng fake và real trong file {} :".format(self.path))
        sr = self.data_from_path['nhan'].value_counts()
        indexes = sr.index
        values = sr.values
        result = ""
        for index in range(len(indexes)):
            if indexes[index] == 1:
                result = result + "so bao fake la : " + str(values[index]) + "    "
            else :
                result = result + "so bao real la : " + str(values[index]) + "    "
        print(result)

    def removeSpecialRow(self):
        noi_dung = self.data_from_path['noi_dung']
        self.data_from_path['noi_dung'] = self.data_from_path['noi_dung'].astype(str)

        self.data_from_path = self.data_from_path[self.data_from_path['noi_dung'] == noi_dung]
