# KLUE Relation Extraction Competition, Naver Boostcamp AI Tech 2๊ธฐ
## Competition Abstract
๐ค KLUE RE(Relation Extraction) Dataset์ผ๋ก ์ฃผ์ด์ง ๋ฌธ์ฅ์ ์ง์ ๋ ๋ Entity์ ๊ด๊ณ๋ฅผ ์ถ์ถ, ๋ถ๋ฅํ๋ Task.  
๐ค Public, Private ๋ฐ์ดํฐ๊ฐ ๋ถ๋ฆฌ๋ Leaderboard ํ๊ฐ๊ฐ ์ด๋ฃจ์ด์ง.  
๐ค ํ๋ฃจ 10ํ๋ก ๋ชจ๋ธ ์ ์ถ ์ ํ

## [Team Portfolio](https://naem1023.notion.site/KLUE-ad3b884f6c2a4f28a00b548aa12c51b6)
## [Solution Presentation(PDF)](competition_results/KLUE_2๋ฑ_์๋ฃจ์(9์กฐ)๋ฐํ์๋ฃ.pdf)
## [Competition Report(PDF)](competition_results/Competition%20Report.pdf)
## Our solutions
- 'klue/roberta-large' with BiLSTM
- Modify Input format
  - Typed Entity Marker with Punctuation 
  - Add Query like Question and Answering
- Augmentation
  - Subject & Object Entity Random Masking
  - AEDA
  - Random Delete
  - Entity swap
- Ensemble
  - Stratified K-Fold & OOF(Out-of-Fold) Prediction
  - K-fold Ensemble via weighted soft voting

## ์ต์ข ์์ 2๋ฑ!
<img src="competition_results/capture.png" width="80%">

--- 
## Docs 
- Model docs
  - [Developed models](models\README.md)
- Augmentation docs
  - [Task Adaptive Pre-Training via Back translation](./augmentation/back_trans/README.md)
  - [EDA](./augmentation/README.md#eda)
  - [AEDA](./augmentation/README.md#aeda)
  - [Random masking](./augmentation/README.md#random-masking)

## Quickstart
### Installation
```
pip install -r requirements.txt
```
### Train model
```python
# default wandb setting in train.py
run = wandb.init(project= 'klue', entity= 'quarter100', name= f'KFOLD_{fold}_{args.wandb_path}')
```

```
python train.py
```
Models are saved in "./best_model/".
### Inference
```
python inference_fold.py
```
Prediction csv files are saved in "./prediction".
### Ensemble
```
python vote.py
```
Ensemble result is saved in "./prediction/submission_fold_total.csv".
