import os
import json
import random
import pandas as pd
from data_controller import DataController
if __name__ == '__main__': 
    loai = "y_te"
    path = f'../dataset/Fake_real_dataset/json/{loai}/fake'
    path_to_save = "../dataset/csv/fake_news.csv"

    print("Ready to read data from path !!!")

    list_articles_dict = []
    for fn in os.listdir(path):
        fpath = f'{path}/{fn}'
        with open(fpath, 'r', encoding='utf-8') as f:
            article_dict = json.load(f)
        list_articles_dict.append(article_dict)

    random.shuffle(list_articles_dict)

    df = pd.DataFrame(list_articles_dict, columns=['loai', 'nguon', 'link', 'tg_dang_tin',
                                                    'tieu_de', 'noi_dung', 'nhan'])
    print("Read successly !!!")

    if DataController.checkFileIsExist(path_to_save):
        data_before = pd.read_csv(path_to_save)
        df = pd.concat([data_before, df], ignore_index=True, sort=False)
        print("Append with data exsit !!!")

    df.to_csv(path_to_save, index=False)
    print("saved data to file csv")
