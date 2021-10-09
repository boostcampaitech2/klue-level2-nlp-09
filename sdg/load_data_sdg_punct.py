import pandas as pd
import torch

class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels
  #출력 : { 'input_ids' : tokenizer를 거친 각 토큰에 대한 사전 속 id를 모은 텐서 리스트 , 
  #        'attention_mask' : 패딩에 해당하는 부분은 0, 나머지는 1을 주어 구분하는 목적의 텐서 리스트, 
  #        'labels' : 데이터의 정답 레이블, 
  #        'start_sub_idx' : tokenizer를 거친 후 Subject entity marker @ token 의 위치 index 
  #        'start_obj_idx' : tokenizer를 거친 후 Object entity marker # token 의 위치 index}
  # Punctuation token들의 tokenizer vocab 속 id : @ -> 36, * -> 14, # -> 7, ^ -> 65 
  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    sub_i = []
    obj_i = []
    for idx_t in range(len(item['input_ids'])-1):    
      if item['input_ids'][idx_t]==36 and item['input_ids'][idx_t+1]==14: # @:36 * : 14
        sub_i.append(idx_t)
      elif item['input_ids'][idx_t]==7 and item['input_ids'][idx_t+1]==65: # # : 7, ^ : 65
        obj_i.append(idx_t)
    item['start_sub_idx'] = sub_i[1]
    item['start_obj_idx'] = obj_i[1]
    return item

  def __len__(self):
    return len(self.labels)

def preprocessing_dataset(data):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  sub_entity, sub_type= [], []
  obj_entity, obj_type= [], []
  sub_idx, obj_idx= [], []
  sentence = []
  
  for i, [x, y, z] in enumerate(zip(data['subject_entity'], data['object_entity'], data['sentence'])):
      x = x.replace('*', '')
      y = y.replace('*', '')
      z = z.replace('*', '')  
      # Subject_entity, Object_entity string에서 Entity type 추출
      # PER, ORG, LOC, DAT, POH, NOH
      sub_typ= x[1:-1].split(':')[-1].split('\'')[-2] 
      obj_typ= y[1:-1].split(':')[-1].split('\'')[-2]
      
      # Subject_entity, Object_entity string에서 문장(Sequence) 속 Entity의 시작, 끝 idx 추출
      # sub_start : Subject_entity의 시작 idx / sub_end : Subject_entity의 끝 idx / obj_start : Object_entity의 시작 idx / obj_end : Object_entity의 끝 idx 
      for idx_i in range(len(x)):
        if x[idx_i: idx_i+ 9]== 'start_idx':
          sub_start= int(x[idx_i+12:].split(',')[0].strip())
        if x[idx_i: idx_i+7]== 'end_idx':
          sub_end= int(x[idx_i+10:].split(',')[0].strip())
        
        if y[idx_i: idx_i+ 9]== 'start_idx':
          obj_start= int(y[idx_i+12:].split(',')[0].strip())
        if y[idx_i: idx_i+7]== 'end_idx':
          obj_end= int(y[idx_i+10:].split(',')[0].strip())
      
      # Entity idx를 모은 리스트
      sub_i= [sub_start, sub_end]
      obj_i= [obj_start, obj_end]
      
      #*_type : 추출한 Entity Type를 리스트에 추가하기
      #*_entity : 추출한 idx 리스트를 통해 Entity 문자열을 리스트에 저장
      sub_entity.append(z[sub_i[0]: sub_i[1]+1])
      obj_entity.append(z[obj_i[0]: obj_i[1]+1])
      sub_type.append(sub_typ); sub_idx.append(sub_i)
      obj_type.append(obj_typ); obj_idx.append(obj_i)
      
      #Typed entity marker(punct) -> An Improved Baseline for Sentence-level Relation Extraction(Zhou, 2021)
      # Ex) Bill was born in Seattle (Subject Entity(Type : PER) : Bill , Object Entity(Type : LOC) : Seattle) -> @ * PER * Bill @ was born in # ^ LOC ^ Seattle #
      if sub_i[0] < obj_i[0]:
        z= z[:sub_i[0]] + '@*'+sub_typ+'*'+ z[sub_i[0]: sub_i[1]+1] + '@' + z[sub_i[1]+1:]
        z= z[:obj_i[0]+7] + '#^'+ obj_typ +'^'+ z[obj_i[0]+7: obj_i[1]+8]+ '#'+ z[obj_i[1]+8:]
      else:
        z= z[:obj_i[0]] + '#^'+ obj_typ +'^'+ z[obj_i[0]: obj_i[1]+1]+ '#' + z[obj_i[1]+1:]
        z= z[:sub_i[0]+7] + '@*'+sub_typ+'*' + z[sub_i[0]+7: sub_i[1]+8] + '@' + z[sub_i[1]+8:]


      sentence.append(z)

  # Dataframe 형태로 저장
  df= pd.DataFrame({'id': data['id'], 'sentence' : sentence, 'subject_entity': sub_entity, 'object_entity': obj_entity,
                          'subject_type': sub_type, 'object_type': obj_type, 'label': data['label'],
                          'subject_idx': sub_idx, 'object_idx': obj_idx})
  return df


def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)
  
  return dataset

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  #Full sentence 형태로 Feed할 예정
  concat_list=[]
  for sub_type, obj_type, sub_entity, obj_entity in zip(dataset['subject_type'], dataset['object_type'], dataset['subject_entity'], dataset['object_entity']):
    text = '@*' + sub_type + '*' + sub_entity + '@' + '와' + '#^' + obj_type + '^' + obj_entity + '#' + '의 관계'
    concat_list.append(text)
  tokens= ['PER', 'LOC', 'POH', 'DAT', 'NOH', 'ORG']
  tokenizer.add_tokens(tokens) 
  tokenized_sentences = tokenizer(
      concat_list,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      return_token_type_ids=False
      )
  return tokenized_sentences


  
  