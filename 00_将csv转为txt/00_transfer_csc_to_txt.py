# Learners：Sunny
# university：SWUFE
# school：RIEM
# 开发时间：2022-06-07 6:52
import os
import time
from tqdm import tqdm
import pandas as pd
import numpy
#-*- coding: UTF-8 -*-

def save_csv_to_text(filename, csv_name, usecols):
    '''
    读取csv的文件，将指定列转换存到txt文件中，usecols=0 摘要，usecols=1 文章
    '''
    DATA_ROOT=r'C:\Users\Senior Chen\Desktop\Ecmt-ML\Project\实战\NLP-新冠疫情下的情绪分析\00_将csv转为txt'
    data = pd.read_csv(os.path.join(DATA_ROOT, csv_name), usecols=[usecols],encoding='utf-8')
    data_list = data.values.tolist()
    result = []
    for item in data_list:
        result.append(item[0])
    print("start process {}".format(filename))
    start_time = time.time()
    with open(filename, 'w', encoding='utf-8') as f:
        for item in tqdm(result):
            f.write(str(item) + '\n')
    f.close()
    print("cost time {}".format(time.time() - start_time))
    print('save {} done!'.format(filename))
    print("---------------------")


if __name__ == '__main__':
    save_csv_to_text('source.txt','source.csv',4)