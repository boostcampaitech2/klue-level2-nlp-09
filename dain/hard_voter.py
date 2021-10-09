import numpy as np
import pandas as pd
import pickle

def num_to_label(n):
    with open('dict_num_to_label.pkl', 'rb') as f:
        dict_num_to_label = pickle.load(f)
    origin_label = dict_num_to_label[n]
    return origin_label

def to_nparray(s) :
    return np.array(list(map(float, s[1:-1].split(','))))

dir = '/opt/ml/'
path1 = f'{dir}/Mybaseline/prediction/submission_klue_rob.csv' # 가져올 csv 파일 주소 입력 필수!
path2 = f'{dir}/Mybaseline/prediction/submission_xlm_rob.csv'

df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)

df1['probs'] = df1['probs'].apply(lambda x : to_nparray(x)*0.5) + df2['probs'].apply(lambda x : to_nparray(x)*0.5) #가중치 조절 필수! 모든 가중치의 합은 1이 되도록
df1['pred_label'] = df1['probs'].apply(lambda x : num_to_label(np.argmax(x)))
df1['probs'] = df1['probs'].apply(lambda x : str(list(x)))

df1.to_csv(f'{dir}/Mybaseline/prediction/hard_voted/klue_rob+xlm_rob.csv', index=False) #저장될 위치 및 이름 지정 필수!