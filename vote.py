import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn.functional as F

def num_to_label(n):
    with open('dict_num_to_label.pkl', 'rb') as f:
        dict_num_to_label = pickle.load(f)
    origin_label = dict_num_to_label[n]
    return origin_label

def to_nparray(s) :
    return np.array(list(map(float, s[1:-1].split(','))))

dir = '/opt/ml/'
path1 = f'{dir}/code/prediction/submission0.csv' # 가져올 csv 파일 주소 입력 필수!
path2 = f'{dir}/code/prediction/submission1.csv'
path3 = f'{dir}/code/prediction/submission2.csv'
path4 = f'{dir}/code/prediction/submission3.csv'
path5 = f'{dir}/code/prediction/submission4.csv'


df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)
df3 = pd.read_csv(path3)
df4 = pd.read_csv(path4)
df5 = pd.read_csv(path5)

add_weight = [0]
for i in range(29):
    add_weight.append(0.05)
    
add_weight_Flag = True

df1['probs'] = df1['probs'].apply(lambda x : to_nparray(x)*0.2) + df2['probs'].apply(lambda x : to_nparray(x)*0.2) + df3['probs'].apply(lambda x : to_nparray(x)*0.2) + df4['probs'].apply(lambda x : to_nparray(x)*0.2) + df5['probs'].apply(lambda x : to_nparray(x)*0.2) 
#가중치 조절 필수! 모든 가중치의 합은 1이 되도록
for i in range(len(df1['probs'])):
    df1['probs'][i] = F.softmax(torch.tensor(df1['probs'][i]), dim=0).detach().cpu().numpy()
    if add_weight_Flag:
        df1['probs'][i]+= add_weight
        df1['probs'][i] = F.softmax(torch.tensor(df1['probs'][i]), dim=0).detach().cpu().numpy()
# 0 제외 나머지 클래스에 0.1씩 더하기?
df1['pred_label'] = df1['probs'].apply(lambda x : num_to_label(np.argmax(x)))
df1['probs'] = df1['probs'].apply(lambda x : str(list(x)))

df1.to_csv(f'{dir}/code/prediction/submission_kfold2_test.csv', index=False) #저장될 위치 및 이름 지정 필수!