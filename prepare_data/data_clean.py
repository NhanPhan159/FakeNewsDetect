import pandas as pd
import re
import numpy as np
from underthesea import word_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud
class DataClean:
    def __init__(self,path=None,path_to_save=None,stopword_vn=None,stopword_en=None,mode="vn"):
        self.path = path
        self.data_frame = pd.DataFrame()
        self.stopword_vn = stopword_vn
        self.stopword_en = stopword_en
        self.sentences = np.array([])
        self.labels = np.array([])
        self.sentences_clean = np.array([])
        self.path_to_save = path_to_save
        self.mode = mode

    def setPath(self,path):
        self.path = path

    def readDataFromPath(self):
        data_read_from_path = pd.read_csv(self.path)
        self.data_frame = pd.concat([self.data_frame,data_read_from_path],ignore_index=True)

    def getFreq(self):
        if self.mode == "vn":
            column = "noi_dung"
        else: column = "text"
        plt.figure(figsize = (20,20))
        Wc = WordCloud(max_words = 200 , width = 1600 , height = 800,
                    min_word_length=5).generate(" ".join(self.data_frame[column]))

        plt.axis("off")
        plt.imshow(Wc , interpolation = 'bilinear')   
        plt.show()
     
    def reorderData(self):
        rate = [0.6,0.2,0.2]
        data_chinhtri = self.data_frame[self.data_frame['loai']=="chinh_tri"]
        data_yte = self.data_frame[self.data_frame['loai']=="y_te"]
        data_xahoi = self.data_frame[self.data_frame['loai']=="xa_luan"]
        data = [data_chinhtri,data_yte,data_xahoi]
        len = [data_chinhtri.shape[0],data_yte.shape[0],data_xahoi.shape[0]]
        start = [0,0,0]
        final = pd.DataFrame()
        for j in rate:
            for i in range(3):
                temp = data[i].iloc[start[i] : start[i]+int(j*len[i])]
                final = pd.concat([final,temp],ignore_index=True)
                start[i] = start[i]+int(j*len[i])
        self.data_frame = final

    def shuffleData(self):
        self.data_frame = self.data_frame.sample(frac=1).reset_index(drop=True)

    def dataFrameToArray(self):
        if self.mode == "vn":
            column_1 = "noi_dung"
            column_2 = "nhan"
        else: 
            column_1 = "text"
            column_2 = "label"
        self.sentences = self.data_frame[column_1].values
        self.labels = self.data_frame[column_2].values

    def removePunctuationMark(self,text):
        return re.sub("\W"," ",text)
    
    def removeWhiteSapce(self,text):
        return re.sub(' +', ' ', text)
    
    def removeNumber(self,text):
        return re.sub("\d","",text)
    
    def removeUnderScore(self,text):
        text = text.replace("_","")
        return text
    
    def lower(aelf,text):
        return text.lower()
     
    def removeStopWord(self,text):
        text_after_remove = " ".join([j for j in text.split() if j not in self.stopword_vn])
        return text_after_remove
    
    def removeStopWordEn(self,text):
        text_after_remove = " ".join([j for j in text.split() if j not in self.stopword_en])
        return text_after_remove
    
    def tokenize(self,text):
        return word_tokenize(text,format="text")
     
    def remove_n_t(self,text):
        text = text.replace("\n","")
        return text.replace("\t","")
    
    def preprocessingVN(self,sentence):
        # sentences_after_preprocessing = []
        # for sentence in self.sentences:
        sentence = self.lower(sentence)
        sentence = self.remove_n_t(sentence)
        sentence = self.removeUnderScore(sentence)
        sentence = self.tokenize(sentence)
        sentence = self.removeNumber(sentence)
        sentence = self.removePunctuationMark(sentence)
        sentence = self.removeWhiteSapce(sentence)
        sentence = self.removeStopWord(sentence)
        #     sentences_after_preprocessing.append(sentence)
        # self.sentences_clean = np.array(sentences_after_preprocessing)
        # print("preprocessing xong !!!")
        return sentence
    
    def preprocessingEN(self,sentence):
        sentence = self.lower(sentence)
        sentence = self.remove_n_t(sentence)
        sentence = self.removeUnderScore(sentence)
        sentence = self.removeNumber(sentence)
        sentence = self.removePunctuationMark(sentence)
        sentence = self.removeWhiteSapce(sentence)
        sentence = self.removeStopWordEn(sentence)
        return sentence
        
    def preprocessing(self):
        sentences_after_preprocessing = []
        if self.mode == 'vn':
            for sentence in self.sentences:
                sentence = self.preprocessingVN(sentence)
                sentences_after_preprocessing.append(sentence)
        else:
            for sentence in self.sentences:
                sentence = self.preprocessingEN(sentence)
                sentences_after_preprocessing.append(sentence)
        self.sentences_clean = np.array(sentences_after_preprocessing)
        print("preprocessing xong !!!")

    def staticsData(self):
        print("Thống kê số lượng fake và real trong file {} :".format(self.path))
        sr = self.data_frame["nhan"].value_counts()
        indexes = sr.index
        values = sr.values
        result = ""
        for index in range(len(indexes)):
            if indexes[index] == 1:
                result = result + "so bao fake la : " + str(values[index]) + "    "
            else :
                result = result + "so bao real la : " + str(values[index]) + "    "
        print(result)

    def exportCSV(self):
        data = [list(a) for a in zip(self.sentences_clean,self.labels)]
        columns = ["noi_dung","label"]
        data_frame = pd.DataFrame(data=data,
                                  columns=columns)
        data_frame.to_csv(self.path_to_save,index=False)
        print("Saved data clean to file")
    
        


