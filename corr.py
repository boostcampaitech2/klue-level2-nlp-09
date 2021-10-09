import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
pd.set_option('display.max_row', 500)
pd.set_option('display.max_columns', 100)

def label_to_num(label):
    num_label = []  # 숫자로 된 label 담을 변수

    with open('dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)

        for val in label:
            num_label.append(dict_label_to_num[val])

    return num_label

data= []

print(os.listdir('./csv'))
for file in os.listdir('./csv'):
    if file[-3:] != 'csv':
        continue
    tmp_data= pd.read_csv(os.path.join('./csv', file))
    tmp_label= label_to_num(tmp_data['pred_label'].values)
    data.append(tmp_label)
df= pd.DataFrame(data).T

print(df.corr())


