# Augmentation ์ฌ์ฉ๋ฒ

## ๐ค [EDA](#eda)  
## ๐ค [AEDA](#aeda)
## ๐ค [Back Translation](back_trans/README.md)
## ๐ค [Random masking](#random-masking)

## EDA  
### Random swap, Random Delete
- [eda.py](./eda.py)๋ฅผ [train.py](../train.py)์ ๊ฐ์ ํด๋์ ๋ฃ์ด์ฃผ์ธ์
- [train.py](../train.py) ์ import๋ฅผ ํด์ฃผ์ธ์  
```py
from eda import *
```
- [train.py](../train.py) ์ eda ํจ์๋ฅผ ์ถ๊ฐํฉ๋๋ค. 
```python
   def eda(dataset):
       dataset = calculate_idx(dataset)
       dataset = random_delete(dataset,0.3)
       return dataset
```
- p๋ augmentation์ด ์ผ์ด๋  ํ๋ฅ ์๋๋ค.  
- swap์ด๋ deleteํ๊ธฐ์ ์ calculate_idxํจ์๋ฅผ ์คํํด์ผ ํฉ๋๋ค.  
- `train_dataset` ์ ์ธ ์ดํ์ `augmented_train_dataset` ์ ์ ์ธํด์ train_dataset ๋์ ์ ์ฌ์ฉํ์๋ฉด ๋ฉ๋๋ค.
```python
augmented_train_dataset = easy_data_augmentation(train_dataset)
```


---
## AEDA
- [aeda.py](./aeda.py) ๋ฅผ [train.py](../train.py)์ ๊ฐ์ ํด๋์ ๋ฃ์ด์ฃผ์ธ์
- [train.py](../train.py) ์ import๋ฅผ ํด์ฃผ์ธ์ 
```py
from aeda import *
```
- [train.py](../train.py) ์ argparse์ ์ ์ฝ๋๋ฅผ ์ถ๊ฐํ๋ฉด aeda๋ก ๋ฌธ์ฅ์ ๋ช๋ฐฐ๋ก ๋๋ฆด ๊ฒ์ธ์ง ์ค์  ๊ฐ๋ฅํฉ๋๋ค.
```py
parser.add_argument("--aeda", type=int, default=2, help="aeda num (default: 2")
``` 
- `train_label` ์ ์ธ ์ดํ์ aeda ์ฝ๋๋ฅผ ์ถ๊ฐํด ์ฃผ๋ฉด ์ฌ์ฉ ๊ฐ๋ฅํฉ๋๋ค.
```py
# data augmentation (AEDA)
if args.aeda > 1:
   train_dataset, train_label = start_aeda(train_dataset, train_label, args.aeda)
```

---
## Random masking
- 2๊ฐ์ง ๋ฐฉ๋ฒ์ผ๋ก random masking ์ฌ์ฉ ๊ฐ๋ฅ
```python
random_maksing/train.py

tokenized_train, token_size= preprocess.tokenized_dataset(train_dataset, tokenizer, mask_flag=True)
tokenized_val, _= preprocess.tokenized_dataset(val_dataset, tokenizer, mask_flag=False)
```
```python
random_maksing/train.py

masked_train = random_masking(tokenized_train, p = 0.15)
```
