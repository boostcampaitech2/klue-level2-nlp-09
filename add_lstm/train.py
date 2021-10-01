import os
import pandas as pd
import torch
import sklearn
import random
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import Trainer,AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TrainingArguments
from sklearn.model_selection import train_test_split, StratifiedKFold
import argparse

import wandb
from model import *
from load_data import *

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


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

  seed_everything(42)
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(device)

  MODEL_NAME= args.model

  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  model= Model(MODEL_NAME)
#  model_config= AutoConfig.from_pretrained(MODEL_NAME)
#  model_config.num_labels= 30
#  model= AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config= model_config)

  special_tokens_dict = {'additional_special_tokens': ['[SUB]','[/SUB]', '[OBJ]', '[/OBJ]']}
  tokenizer.add_special_tokens(special_tokens_dict)
  model.model.resize_token_embeddings(tokenizer.vocab_size + 4)
#  model.resize_token_embeddings(tokenizer.vocab_size + 4)


  all_dataset= load_data("../dataset/train/train.csv")
  all_label= label_to_num(all_dataset['label'].values)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(device)


  train_data, val_data, train_label, val_label= train_test_split(all_dataset, all_label, test_size= 0.2, stratify= all_label)

  # train_label= label_to_num['label'].values
  # val_label

  tokenized_train= tokenized_dataset(train_data, tokenizer)
  tokenized_val= tokenized_dataset(val_data, tokenizer)

  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  RE_val_dataset = RE_Dataset(tokenized_val, val_label)


  save_dir= f'./result/{MODEL_NAME}_add_lstm'

  training_args = TrainingArguments(
          output_dir=save_dir,        # output directory
          save_total_limit=2,              # number of total save model.
          save_steps=args.save_steps,                 # model saving step.
          num_train_epochs=args.epochs,              # total number of training epochs
          learning_rate=args.lr,               # learning_rate
          # batch size per device during training
          per_device_train_batch_size=args.batch,
          per_device_eval_batch_size=args.batch_valid,   # batch size for evaluation
          # number of warmup steps for learning rate scheduler
          label_smoothing_factor = 0.1,
          warmup_steps=args.warmup,
          weight_decay=args.weight_decay,               # strength of weight decay
          logging_dir='./logs',            # directory for storing logs
          logging_steps=args.logging_steps,              # log saving step.
          metric_for_best_model=args.metric_for_best_model,
          evaluation_strategy='steps',  # evaluation strategy to adopt during training
          # `no`: No evaluation during training.
          # `steps`: Evaluate every `eval_steps`.
          # `epoch`: Evaluate every end of epoch.
          eval_steps=args.eval_steps,            # evaluation step.
          load_best_model_at_end=True
      )
  
  trainer = Trainer(
          # the instantiated 🤗 Transformers model to be trained
          model=model,
          args=training_args,                  # training arguments, defined above
          train_dataset=RE_train_dataset,         # training dataset
          eval_dataset=RE_val_dataset,             # evaluation dataset
          compute_metrics=compute_metrics         # define metrics function
      )


  run = wandb.init(project='klue', entity='quarter100', name='seokmin/roberta/normal_sep')
  trainer.train()
  model.save_pretrained('./best_model'+'_lstm')
  run.finish()

def main():
  train()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--seed', type=int, default=42,
                      help='random seed (default: 42)')
  parser.add_argument('--fold', type=int, default=5,
                      help='fold (default: 5)')
  parser.add_argument('--model', type=str, default='klue/roberta-large',
                      help='model type (default: xlm-roberta-large)')
  parser.add_argument('--epochs', type=int, default=5,
                      help='number of epochs to train (default: 5)')
  parser.add_argument('--lr', type=float, default=1e-5,
                      help='learning rate (default: 5e-5)')
  parser.add_argument('--batch', type=int, default=16,
                      help='input batch size for training (default: 16)')
  parser.add_argument('--batch_valid', type=int, default=16,
                      help='input batch size for validing (default: 16)')
  parser.add_argument('--warmup', type=int, default=200,
                      help='warmup_steps (default: 200)')
  parser.add_argument('--eval_steps', type=int, default=406,
                      help='eval_steps (default: 406)')
  parser.add_argument('--save_steps', type=int, default=406,
                      help='save_steps (default: 406)')
  parser.add_argument('--logging_steps', type=int,
                      default=100, help='logging_steps (default: 100)')
  parser.add_argument('--weight_decay', type=float,
                      default=0.01, help='weight_decay (default: 0.01)')
  parser.add_argument('--metric_for_best_model', type=str, default='micro f1 score',
                      help='metric_for_best_model (default: micro f1 score')

  args = parser.parse_args()
  print(args)
  main()
