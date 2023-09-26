from stop_word_vn import STOP_WORDS
from data_clean import DataClean
import pandas as pd
import numpy as np

if __name__ == '__main__':
    ## rate train/validate/test
    rate = [0.6,0.2,0.2]
    ###
    path_real = "../dataset/csv/real_news_viet.csv"
    path_fake = "../dataset/csv/fake_news_viet.csv"
    paths_save = ["../dataset/preprocessing/train_vn.csv",
                  "../dataset/preprocessing/validate_vn.csv",
                  "../dataset/preprocessing/test_vn.csv"]

    # read data real, fake và làm sạch
    data_cleaner_real = DataClean(path=path_real, path_to_save="", stopword_vn=STOP_WORDS,mode='vn')
    data_cleaner_real.readDataFromPath()
    data_cleaner_real.reorderData() ### phân bố lại dữ liệu
    data_cleaner_real.dataFrameToArray()
    data_cleaner_real.preprocessing()

    data_cleaner_fake = DataClean(path=path_fake, path_to_save="", stopword_vn=STOP_WORDS,mode='vn')
    data_cleaner_fake.readDataFromPath()
    data_cleaner_fake.reorderData() ### phân bố lại dữ liệu
    data_cleaner_fake.dataFrameToArray()
    data_cleaner_fake.preprocessing()

###
    # sr = data_cleaner_fake.data_frame["loai"].iloc[int(1133*0.6)+int(1133*0.2):].value_counts()
    # print(sr)
    # sr = data_cleaner_real.data_frame["loai"].iloc[int(1208*0.6)+int(1208*0.2):].value_counts()
    # print(sr)
    # sr = data_cleaner_fake.data_frame["loai"].iloc[0:int(1133*0.6)].value_counts()
    # print(sr)
    # sr = data_cleaner_real.data_frame["loai"].iloc[0:int(1208*0.6)].value_counts()
    # print(sr)
###
    # chia data theo ti lệ
    len_real = len(data_cleaner_real.sentences_clean)
    len_fake = len(data_cleaner_fake.sentences_clean)

    start_real = 0
    start_fake = 0
    for i in range(len(paths_save)):
        end_real = start_real + int(rate[i] * len_real)
        end_fake = start_fake + int(rate[i] * len_fake)

        sentences_real = data_cleaner_real.sentences_clean[start_real:end_real]
        sentences_fake = data_cleaner_fake.sentences_clean[start_fake:end_fake]
        sentences = np.concatenate((sentences_real, sentences_fake), axis=0)

        labels_real = data_cleaner_real.labels[start_real:end_real]
        labels_fake = data_cleaner_fake.labels[start_fake:end_fake]
        labels = np.concatenate((labels_real, labels_fake), axis=0)

        data = [list(a) for a in zip(sentences,labels)]
        columns = ["noi_dung","nhan"]
        data_frame = pd.DataFrame(data=data,
                                  columns=columns)
        data_frame = data_frame.replace(r'^s*$', float('NaN'), regex = True)
        data_frame.dropna(inplace=True)
        data_frame = data_frame.sample(frac=1).reset_index(drop=True)
        data_frame.to_csv(paths_save[i],index=False)
        print("Saved data clean to file : ",paths_save[i])

        start_fake = end_fake
        start_real = end_real
    

    




