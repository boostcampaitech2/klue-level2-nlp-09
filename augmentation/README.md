# Augmentation 사용법

## 🤗 [EDA](#eda)  
## 🤗 [AEDA](#aeda)
## 🤗 [Back Translation](back_trans/README.md)


## EDA  
### Random swap, Random Delete
- [eda.py](./eda.py)를 [train.py](../train.py)와 같은 폴더에 넣어주세요
- [train.py](../train.py) 에 import를 해주세요  
```py
from eda import *
```
- [train.py](../train.py) 에 eda 함수를 추가합니다. 
```python
   def eda(dataset):
       dataset = calculate_idx(dataset)
       dataset = random_delete(dataset,0.3)
       return dataset
```
- p는 augmentation이 일어날 확률입니다.  
- swap이나 delete하기전에 calculate_idx함수를 실행해야 합니다.  
- `train_dataset` 선언 이후에 `augmented_train_dataset` 을 선언해서 train_dataset 대신에 사용하시면 됩니다.
```python
augmented_train_dataset = easy_data_augmentation(train_dataset)
```


---
## AEDA
- [aeda.py](./aeda.py) 를 [train.py](../train.py)와 같은 폴더에 넣어주세요
- [train.py](../train.py) 에 import를 해주세요 
```py
from aeda import *
```
- [train.py](../train.py) 의 argparse에 위 코드를 추가하면 aeda로 문장을 몇배로 늘릴 것인지 설정 가능합니다.
```py
parser.add_argument("--aeda", type=int, default=2, help="aeda num (default: 2")
``` 
- `train_label` 선언 이후에 aeda 코드를 추가해 주면 사용 가능합니다.
```py
train_label = preprocess.label_to_num(train_dataset["label"].values)
val_label = preprocess.label_to_num(val_dataset["label"].values)

# data augmentation (AEDA)
if args.aeda > 1:
   train_dataset, train_label = start_aeda(train_dataset, train_label, args.aeda)
```


## Random masking
```python
train.py

tokenized_train, token_size= preprocess.tokenized_dataset(train_dataset, tokenizer, mask_flag=True)
tokenized_val, _= preprocess.tokenized_dataset(val_dataset, tokenizer, mask_flag=True)
```
