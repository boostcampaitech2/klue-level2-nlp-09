import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from konlpy.tag import Mecab

import os
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast
import pandas as pd

import pickle as pickle
import os
import pandas as pd
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import re

"""
TOTAL TRAIN SIZE : 32470

"""

class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class. 우리가 모델에 데이터를 먹여주기 위해서는 dataset으로 만들어줘야 한다 !""" 
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

def load_data(data):
  
  sub_entity, sub_type= [], []
  obj_entity, obj_type= [], []
  sub_idx, obj_idx= [], []
  sentence= []

  for i, [x, y, z] in enumerate(zip(data['subject_entity'], data['object_entity'], data['sentence'])):
      sub_typ= x[1:-1].split(':')[-1].split('\'')[-2]
      obj_typ= y[1:-1].split(':')[-1].split('\'')[-2]
      
      for idx_i in range(len(x)):
        if x[idx_i: idx_i+ 9]== 'start_idx':
          sub_start= int(x[idx_i+12:].split(',')[0].strip())
        if x[idx_i: idx_i+7]== 'end_idx':
          sub_end= int(x[idx_i+10:].split(',')[0].strip())
        
        if y[idx_i: idx_i+ 9]== 'start_idx':
          obj_start= int(y[idx_i+12:].split(',')[0].strip())
        if y[idx_i: idx_i+7]== 'end_idx':
          obj_end= int(y[idx_i+10:].split(',')[0].strip())
      
      sub_i= [sub_start, sub_end]
      obj_i= [obj_start, obj_end]

      sub_entity.append(z[sub_i[0]: sub_i[1]+1])
      obj_entity.append(z[obj_i[0]: obj_i[1]+1])
      sub_type.append(sub_typ); sub_idx.append(sub_i)
      obj_type.append(obj_typ); obj_idx.append(obj_i)

      if sub_i[0] < obj_i[0]:
        z= z[:sub_i[0]] + '[SUB]'+ z[sub_i[0]: sub_i[1]+1] + '[SUB]' + z[sub_i[1]+1:]
        z= z[:obj_i[0]+10] + '[OBJ]'+ z[obj_i[0]+10: obj_i[1]+11]+ '[OBJ]'+ z[obj_i[1]+15:]
      else:
        z= z[:obj_i[0]] + '[OBJ]'+ z[obj_i[0]: obj_i[1]+1]+ '[OBJ]'+ z[obj_i[1]+1:]
        z= z[:sub_i[0]+10] + '[SUB]'+ z[sub_i[0]+10: sub_i[1]+11] + '[SUB]' + z[sub_i[1]+15:]


      sentence.append(z)

  df= pd.DataFrame({'id': data['id'], 'sentence' : sentence, 'subject_entity': sub_entity, 'object_entity': obj_entity,
                          'subject_type': sub_type, 'object_type': obj_type, 'label': data['label'],
                          'subject_idx': sub_idx, 'object_idx': obj_idx})

#  for i in range(10):
#    print(f"SUB : {df.loc[i]['subject_entity']}\nOBJ : {df.loc[i]['object_entity']}\nSENTENCE : {df.loc[i]['sentence']}\n\n")
  
  return df



if __name__ == '__main__':

  TRAIN_PATH= '/opt/ml/dataset/train/train.csv'

  data= pd.read_csv(TRAIN_PATH)
  load_data(data)
  # print(preprocess.df)
  # preprocess.save_tokenizing()
  # preprocess.tokenized_dataset()
  
