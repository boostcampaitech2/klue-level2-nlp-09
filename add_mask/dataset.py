import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer
import pickle

class Preprocess:
    def __init__(self, CSV_PATH, version):
        self.version= version
        self.data= self.load_data(CSV_PATH)

    def load_data(self, path):
        
        data= pd.read_csv(path)
        
        sub_entity, sub_type= [], []
        obj_entity, obj_type= [], []
        sub_idx, obj_idx= [], []
        sentence= []

        for i, [x, y, z] in enumerate(zip(data['subject_entity'], data['object_entity'], data['sentence'])):
            
            x = x.replace('*', '')
            y = y.replace('*', '')
            z = z.replace('*', '')
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
            
            if self.version== 'SUB':
                if sub_i[0] < obj_i[0]:
                    z= z[:sub_i[0]] + '[SUB]'+ z[sub_i[0]: sub_i[1]+1] + '[/SUB]' + z[sub_i[1]+1:]
                    z= z[:obj_i[0]+11] + '[OBJ]'+ z[obj_i[0]+11: obj_i[1]+12]+ '[/OBJ]'+ z[obj_i[1]+12:]
                else:
                    z= z[:obj_i[0]] + '[OBJ]'+ z[obj_i[0]: obj_i[1]+1]+ '[/OBJ]'+ z[obj_i[1]+1:]
                    z= z[:sub_i[0]+11] + '[SUB]'+ z[sub_i[0]+11: sub_i[1]+12] + '[/SUB]' + z[sub_i[1]+12:]

            elif self.version== 'PUN':
                if sub_i[0] < obj_i[0]:
                    z= z[:sub_i[0]] + '@*'+sub_typ+'*'+ z[sub_i[0]: sub_i[1]+1] + '@' + z[sub_i[1]+1:]
                    z= z[:obj_i[0]+7] + '#^'+ obj_typ +'^'+ z[obj_i[0]+7: obj_i[1]+8]+ '#'+ z[obj_i[1]+8:]
                else:
                    z= z[:obj_i[0]] + '#^'+ obj_typ +'^'+ z[obj_i[0]: obj_i[1]+1]+ '#' + z[obj_i[1]+1:]
                    z= z[:sub_i[0]+7] + '@*'+sub_typ+'*' + z[sub_i[0]+7: sub_i[1]+8] + '@' + z[sub_i[1]+8:]

            sentence.append(z)

        """special token과 type 추가한 데이터 프레임 생성"""
        df= pd.DataFrame({'id': data['id'], 'sentence' : sentence, 'subject_entity': sub_entity, 'object_entity': obj_entity,
                                'subject_type': sub_type, 'object_type': obj_type, 'label': data['label'],
                                'subject_idx': sub_idx, 'object_idx': obj_idx})

        # check add [sub], [obj] token sentence
        # for i in range(10):    
        #     print(f"SUB : {df.loc[i]['subject_entity']}\nOBJ : {df.loc[i]['object_entity']}\nSENTENCE : {df.loc[i]['sentence']}\n\n")
        
        return df
    
    def tokenized_dataset(self, data, tokenizer):
        """add tokens"""
        tokens= ['PER', 'LOC', 'POH', 'DAT', 'NOH', 'ORG']
        tokenizer.add_tokens(tokens)     

        concat_entity = []
        for sub_ent, obj_ent, sub_typ, obj_typ in zip(data['subject_entity'], data['object_entity'], data['subject_type'], data['object_type']):
            temp =  '@*'+ sub_typ + '*' + sub_ent + '@와 #^' + obj_typ + '^' + obj_ent + '#의 관계'
            #temp =  e01 + '와' + e02 + '의 관계'
            concat_entity.append(temp)

        tokenized_sentence= tokenizer(
            concat_entity,
            list(data['sentence']), # list나 string type으로 보내줘야 함 !
            return_tensors= "pt", # pytorch type
            padding= True, # 문장의 길이가 짧다면 padding
            truncation= True, # 문장 자르기
            max_length= 256, # 토큰 최대 길이...
            add_special_tokens= True, # special token 추가
            return_token_type_ids= False # roberta의 경우.. token_type_ids가 안들어감 ! 
        )   

        """결국 모델에 들어가는 input은 tokenized된 문장들이다 !
            tokenized_sentence에는 input_ids, token_type_ids, attention_mask가 있어야 model로 들어간다.
            하지만 단순히 이렇게 tokenize한 데이터로 model에 들어가지 않는다 !
            dataset으로 가공을 해줘야 한다.
        """
        # check decoded, tokenized token
        # for i in range(10):
        #     text_tok= tokenizer.tokenize(concat_entity[i],list(data['sentence'])[i], add_special_tokens=True)

        #     text_enc= tokenizer.encode(concat_entity[i],list(data['sentence'])[i], add_special_tokens=True)
        #     text_dec= tokenizer.decode(text_enc)
        #     print(text_dec)
        #     print(text_tok)
        #     print()       

        return tokenized_sentence, len(tokens)
    
    def label_to_num(self, label):
        num_label= [] # 숫자로 된 label 담을 변수

        with open('dict_label_to_num.pkl', 'rb') as f:
            dict_label_to_num= pickle.load(f)

            for val in label:
                num_label.append(dict_label_to_num[val])
        
        return num_label


"""
    기본적인 dataset은 init, getitem, len기능이 있어야 한다.
    Dataset에 dict type으로 input_ids, attention_mask, labels를 넣어주는 느낌
    이 때 label은 tensor type
"""

class Dataset:
    def __init__(self, data, labels): # data : dict, label : list느낌..
        self.data= data
        self.labels= labels
    
    """dict, list type의 input이 들어왔음.."""
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.data.items()}
        item['labels'] = torch.tensor(self.labels[idx])

        return item
    
    def __len__(self):
        return len(self.labels)



if __name__ == '__main__':
    
    TRAIN_PATH= '/opt/ml/dataset/train/train_revised.csv'
    MODEL_NAME= 'klue/roberta-large'

    preprocess= Preprocess(TRAIN_PATH, "PUN")
    tokenizer= AutoTokenizer.from_pretrained(MODEL_NAME)

    # 아직 Dataset으로 가공을 하지 않았음 ! 단순히 dict와 List임
    all_dataset= preprocess.data
    # all_label= preprocess.label_to_num(all_dataset['label'].values)

    tokenized_dataset= preprocess.tokenized_dataset(preprocess.data, tokenizer)

    # RE_DATASET= Dataset(tokenized_dataset, all_label)
    # print(RE_DATASET[3])



