import pickle as pickle
import os
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import lr_scheduler
from transformers import AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback, get_cosine_with_hard_restarts_schedule_with_warmup
from load_data_sdg_punct import *
import random
from sklearn.model_selection import StratifiedKFold
import argparse
from model import REmodel
import wandb
#랜덤 시드 고정
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
    return f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

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
  f1ver2 = f1_score(labels, preds, average="micro") * 100

  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
      'normal f1' : f1ver2
  }

def label_to_num(label):
  num_label = []
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label



def train():
  seed_everything(args.seed)
  MODEL_NAME = "klue/roberta-large"
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  
  print(device)
  
  default_dataset = load_data("../dataset/train/train_revised.csv")
  default_label = label_to_num(default_dataset['label'].values)
  
  #Kfold 사용
  kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
  
  for fold, (train_idx, val_idx) in enumerate(kfold.split(default_dataset, default_label)):
        print(f"{fold} FOLD")
        run=wandb.init(project='klue', entity='quarter100', name=args.wandb_name+str(fold))            
        train_dataset = default_dataset.iloc[train_idx]
        valid_dataset = default_dataset.iloc[val_idx]
        
        train_label = label_to_num(train_dataset['label'].values)
        valid_label = label_to_num(valid_dataset['label'].values)
        
        # tokenizing dataset
        tokenized_train = tokenized_dataset(train_dataset, tokenizer)
        tokenized_valid = tokenized_dataset(valid_dataset, tokenizer)

        # make dataset for pytorch.
        RE_train_dataset = RE_Dataset(tokenized_train, train_label)
        RE_valid_dataset = RE_Dataset(tokenized_valid, valid_label)

        model =  REmodel(MODEL_NAME, device)
        model.model.resize_token_embeddings(tokenizer.vocab_size + 6)
        model.to(device)
        
        training_args = TrainingArguments(
        output_dir='./results/'+'fold'+str(fold),          # output directory
        save_total_limit=1,              # number of total save model.
        save_steps=125,                 # model saving step.
        num_train_epochs=args.epoch,              # total number of training epochs
        gradient_accumulation_steps=2,       
        learning_rate=args.lr,               # learning_rate
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.batch_size,   # batch size for evaluation
        warmup_ratio=0.1,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=25,              # log saving step.
        evaluation_strategy='steps', # evaluation strategy to adopt during training
                                    # `no`: No evaluation during training.
                                    # `steps`: Evaluate every `eval_steps`.
                                    # `epoch`: Evaluate every end of epoch.
        eval_steps = 125,            # evaluation step.
        load_best_model_at_end = True,
        seed = args.seed,
        group_by_length=True,
        metric_for_best_model=args.metric_for_best_model,
        label_smoothing_factor = 0.1,
        report_to="wandb",
        dataloader_num_workers=2,
        )
        trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=RE_train_dataset,         # training dataset
        eval_dataset=RE_valid_dataset,             # evaluation dataset
        compute_metrics=compute_metrics,         # define metrics function
        callbacks = [EarlyStoppingCallback(early_stopping_patience=args.early_stop)],
        )
        
        # train model
        trainer.train()
        model_object_file_path =  args.save_dir+str(fold)
        if not os.path.exists(model_object_file_path):
          os.makedirs(model_object_file_path)
        torch.save(model.state_dict(), os.path.join(model_object_file_path, 'pytorch_model.bin'))
        run.finish()
        
        
    
def main():
  torch.cuda.empty_cache()
  os.environ["TOKENIZERS_PARALLELISM"] = "false"
  train()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
  parser.add_argument('--epoch', type=int, default=5, help='num epoch (default: 5)')
  parser.add_argument('--lr', type=float, default=3e-5, help='learning_rate (default: 3e-5)')
  parser.add_argument('--batch_size', type=int, default=32, help='batch size (default: 32)')
  parser.add_argument('--metric_for_best_model', type=str, default='micro f1 score', help='metric for best model (default : micro f1 score')
  parser.add_argument('--early_stop', type=int, default=3, help='ealry stop (default: 3)')
  parser.add_argument('--wandb_name', type=str, default='sdg REMODEL kfold', help='name of wandb, will be displayed with fold number i (need to change for each test)')
  parser.add_argument('--save_dir', type=str, default = './best_model/fold', help='save_dir for each fold\'s best model (default : ./best_model/fold i)')
  
  
  
  args =parser.parse_args()
  print(args)
  
  main()
  
    
    
    
    