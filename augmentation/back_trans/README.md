# Back translation for Task Adaptive Pre-training

🤗 Back translation: Papago 번역기를 web crawling.  
🤗 Pre-trianer: HuggingFace의 maksed language model을 Pre-train.    
🤗 Parameters: Best model에 Pre-trained model을 load.
  
## Required Installations
```
pip install -r requirements.txt
```
- chromedriver

## How to use
### Back translation
```
# 기본 사용 방법
python back_translation.py

# nohup으로 log 보고 싶으면
bt.sh
```

**Options**
- --remove_stop_words: remove stop words (default: False)
- --only_kor_to_en: translate only kor to en (default: False)
- --only_en_to_kor: translate only en to kor (default: False)
- --len: specify length of csv file (default: False)

**Outputs**
- final_kor_to_eng_{file_time}.npy
- final_en_to_kor_{file_time}.npy
- back_translation_result.csv: contain kor_to_eng and eng_to_kor.

### Pre-training
```
python pretrain.py
```
Pre-trained model saved dir = './pretrined_model'.

### Load Pre-trained model for our best model
- Import back_trans/parameters.BackTransPreTrain in your model.
- MODEL_NAME: must use 'klue/roberta-large'
    - 'klue/roberta-large'를 transformers.AutoModel나 transformers.AutoModelForMaskedLM로 load해야지만 pre-trained model을 사용 가능.
```python
from back_trans import BackTransPreTrain

model = AutoModel.from_pretrained(MODEL_NAME)

bpt = BackTransPreTrain(pretrain_path)
model.load_state_dict(bpt.load_parameters(MODEL_NAME))
```
