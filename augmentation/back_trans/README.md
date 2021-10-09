# Back translation for Task Adaptive Pre-training

🤗 Back translation: Using Papago translator via web crawling.  
🤗 Pre-trianer: Train maksed language model in HuggingFace.    
🤗 Parameters: Load Pre-trained model for our best model.  
  
## Required Installations
```
pip install -r requirements.txt
```
- chromedriver

## How to use
### Back translation
```
python back_translation.py

# Get log via nohup
bt.sh
```

**Options**
- --remove_stop_words: remove stop words (default: False)
- --only_kor_to_en: translae only kor to en (default: False)
- --only_en_to_kor: translae only en to kor (default: False)
- --len: specify length of csv file (default: False)

**Outputs**
- final_kor_to_eng_{file_time}.npy
- final_en_to_kor_{file_time}.npy
- back_translation_result.csv: contain kor_to_eng and eng_to_kor.

### Pre-training
```
python pretrain.py
```
Pre-trained model will saved at './pretrined_model'.

### Load Pre-trained model for our best model
- Import back_trans/parameters.BackTransPreTrain in your model.
- MODEL_NAME: must use 'klue/roberta-large'
    - Only 'klue/roberta-large' in transformers.AutoModel or transformers.AutoModelForMaskedLM can load pre-trained model.
```python
from back_trans import BackTransPreTrain()

model= AutoModel.from_pretrained(MODEL_NAME)
pretrained = BackTransPreTrain()

bpt = BackTransPreTrain(pretrain_path)
model.load_state_dict(bpt.load_parameters(MODEL_NAME))
```