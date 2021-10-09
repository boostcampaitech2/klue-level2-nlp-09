# Augmentation 사용법

## 🤗 [EDA](#eda)  
## 🤗 [AEDA](#aeda)
## 🤗 [Back Translation](back_trans/README.md)

## EDA(Easy )
### Random swap  
* eda.py 파일 만들고 코드를 복붙합니다.  
* train.py 에서 from eda import.py를 합니다.  
* train.py에 easy_data_augmentation 함수를 추가합니다. 
* easy_data_augmentation 함수안에서 random_delete함수를 실행합니다.  
* p는 augmentation이 일어날 확률입니다.  
* 따라서 random_delete나 random_swap을 하기전에 calculate_idx 함수를 사용하면 문제가 없습니다.(calculate_idx(dataset) -> return dataset)  
* swap이나 delete하기전에 calculate_idx함수를 실행해야 합니다.  
<br>  

```python
   from eda import *

   def eda(dataset):
       dataset = calculate_idx(dataset)
       dataset = random_delete(dataset,0.3)
       return dataset
```
<br>  

* train.py에서 def train 함수 안에서 train_dataset을 load하고 preprocess를 하기전에  
augmented_train_dataset = easy_data_augmented(train_dataset,p) 코드를 추가합니다.  

---
## AEDA
```py
parser.add_argument("--aeda", type=int, default=2, help="aeda num (default: 2")
``` 
- [aeda.py](https://github.com/boostcampaitech2/klue-level2-nlp-09/blob/minji-data/aeda.py) 를 [train.py](https://github.com/boostcampaitech2/klue-level2-nlp-09/blob/minji-data/train.py)와 같은 폴더에 넣어주세요
- [train.py](https://github.com/boostcampaitech2/klue-level2-nlp-09/blob/minji-data/train.py) 의 argparse에 위 코드를 추가하면 aeda로 문장을 몇배로 늘릴 것인지 설정 가능합니다.

```py
train_label = preprocess.label_to_num(train_dataset["label"].values)
val_label = preprocess.label_to_num(val_dataset["label"].values)

# data augmentation (AEDA)
if args.aeda > 1:
   train_dataset, train_label = start_aeda(train_dataset, train_label, args.aeda)
```
- `train_label` 선언 이후에 aeda 코드를 추가해 주면 사용 가능합니다
