import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from load_data import *
import wandb
import random
from pathlib import Path
from pytorch_lightning import seed_everything
from config_parser import JsonConfigFileManager
from sklearn.model_selection import KFold, StratifiedKFold
conf = JsonConfigFileManager('./config.json')
from torch.utils.data import DataLoader
from MyTrainer import *

# https://github.com/ShannonAI/ChineseBert/blob/cbc4a52c7b803189e79367c0b1cf562ef79f21f2/utils/random_seed.py
def set_random_seed(seed: int):
    """set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1(preds, labels)
  auprc = klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.
  
  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }

def label_to_num(label):
  num_label = []
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

def train():
  # check device
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(device)

  # set random seed
  set_random_seed(conf.values['random_seed'])
  
  # set wandb 
  wandb_config = {'test_name': conf.values['test_name'], 'random_seed': conf.values['random_seed'], 'kfold': conf.values['train_settings']['kfold']}
  run = wandb.init(project = conf.values['wandb_options']['project'], entity = conf.values['wandb_options']['entity'], config = wandb_config)
 

  # load model and tokenizer
  MODEL_NAME = conf.values['model_name']
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  

  # load dataset
  default_dataset  = load_data(conf.values['data']['train_data_dir'])
  default_label  = label_to_num(default_dataset['label'].values)

  # K-fold
  kfold = StratifiedKFold(n_splits=conf.values['train_settings']['kfold'], shuffle=True, random_state=conf.values['random_seed'])
  for fold, (train_idx, val_idx) in enumerate(kfold.split(default_dataset, default_label)):
    print(f"{fold} FOLD")

    train_label = label_to_num(default_dataset['label'].iloc[train_idx].values)
    valid_label = label_to_num(default_dataset['label'].iloc[val_idx].values)
    
    train_dataset = default_dataset.iloc[train_idx]
    valid_dataset = default_dataset.iloc[val_idx]

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_valid = tokenized_dataset(valid_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_valid_dataset = RE_Dataset(tokenized_valid, valid_label)
    
    # setting model hyperparameter
    model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30

    model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    #print(model.config)
    model.parameters
    model.to(device)
    
    save_dir = increment_path('./results/'+MODEL_NAME)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
      output_dir = conf.values['huggingface_options']['output_dir'],         # output directory
      save_total_limit = conf.values['huggingface_options']['save_total_limit'],              # number of total save model.
      save_steps = conf.values['huggingface_options']['save_steps'],                 # model saving step.
      num_train_epochs = conf.values['huggingface_options']['num_train_epochs'],              # total number of training epochs
      learning_rate=conf.values['huggingface_options']['learning_rate'],               # learning_rate
      per_device_train_batch_size=conf.values['huggingface_options']['per_device_train_batch_size'],  # batch size per device during training
      per_device_eval_batch_size=conf.values['huggingface_options']['per_device_eval_batch_size'],   # batch size for evaluation
      warmup_steps=conf.values['huggingface_options']['warmup_steps'],                # number of warmup steps for learning rate scheduler
      weight_decay=conf.values['huggingface_options']['weight_decay'],              # strength of weight decay
      logging_dir=conf.values['huggingface_options']['logging_dir'],           # directory for storing logs
      logging_steps=conf.values['huggingface_options']['logging_steps'],              # log saving step.
      evaluation_strategy=conf.values['huggingface_options']['evaluation_strategy'],
      eval_steps = conf.values['huggingface_options']['eval_steps'],
      load_best_model_at_end = conf.values['huggingface_options']['load_best_model_at_end'],
      metric_for_best_model = conf.values['huggingface_options']['metric_for_best_model'],
      #report_to = "wandb"
    )

    trainer = MyTrainer(
      loss_name=conf.values['train_settings']['loss'],
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=RE_train_dataset,         # training dataset
      eval_dataset=RE_valid_dataset,             # evaluation dataset
      compute_metrics=compute_metrics         # define metrics function
    )

    # train model
    trainer.train()
    model.save_pretrained(f'./best_model_{fold}')
    run.finish()
  


def main():
  train()

if __name__ == '__main__':
  main()
