import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from load_data_sdg import *
import random
from sklearn.model_selection import StratifiedKFold
import argparse
from model import REmodel
import wandb
from focal_loss import FocalLoss
from sklearn.utils.class_weight import compute_class_weight

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

class MyTrainer(Trainer):
    def __init__(self, loss_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_name= loss_name

    def compute_loss(self, model, inputs, return_outputs=False):
        
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        
        if self.loss_name == 'CrossEntropy':
            class_weight = compute_class_weight(class_weight = "balanced", classes=np.unique(labels), y=labels)
            custom_loss = torch.nn.CrossEntropyLoss(weight = class_weight).to(device)
            loss = custom_loss(outputs['logits'], labels)
        elif self.loss_name == 'FocalLoss' :
            custom_loss = FocalLoss(gamma=0.5).to(device)
            loss = custom_loss(outputs['logits'], labels)
        elif self.loss_name == 'LabelSmoothLoss' and self.label_smoother is not None:
            loss = self.label_smoother(outputs, labels)
            loss = loss.to(device)
        else:
            print("invalid loss function argument")
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs['logits']
        
        return (loss, outputs) if return_outputs else loss

  
  
def train():
   seed_everything(42)
   MODEL_NAME = "klue/roberta-large"
   tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
   device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
   
   print(device)
   
   default_dataset = load_data("../dataset/train/train_revised.csv")
   default_label = label_to_num(default_dataset['label'].values)
  
   kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
   
   for fold, (train_idx, val_idx) in enumerate(kfold.split(default_dataset, default_label)):
        print(f"{fold} FOLD")
        run=wandb.init(project='klue', entity='quarter100', name='sdg'+'20210931kfold'+'fold'+str(fold))            
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

        # setting model hyperparameter
        
        model =  REmodel(MODEL_NAME, device)
        model.to(device)
        model.model.resize_token_embeddings(tokenizer.vocab_size + 16)
        
        training_args = TrainingArguments(
        output_dir='./results/'+'fold'+str(fold),          # output directory
        save_total_limit=1,              # number of total save model.
        save_steps=250,                 # model saving step.
        num_train_epochs=5,              # total number of training epochs
        learning_rate=3e-5,               # learning_rate
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=32,   # batch size for evaluation
        warmup_steps=406,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=50,              # log saving step.
        evaluation_strategy='steps', # evaluation strategy to adopt during training
                                    # `no`: No evaluation during training.
                                    # `steps`: Evaluate every `eval_steps`.
                                    # `epoch`: Evaluate every end of epoch.
        eval_steps = 250,            # evaluation step.
        load_best_model_at_end = True,
        seed = 42,
        fp16=True,
        group_by_length=True,
        metric_for_best_model='micro f1 score',
        label_smoothing_factor = 0.1,
        report_to="wandb",
        dataloader_num_workers=2
        )
        trainer = MyTrainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=RE_train_dataset,         # training dataset
        eval_dataset=RE_valid_dataset,             # evaluation dataset
        compute_metrics=compute_metrics,         # define metrics function
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
        loss_name = 'LabelSmoothLoss'
        )
        
        # train model
        trainer.train()
        model_object_file_path =  './best_model/'+'fold'+str(fold)+'/pytorch_model.bin'
        if not os.path.exists(model_object_file_path):
          os.makedirs(model_object_file_path)
        torch.save(model.state_dict(), model_object_file_path)
        run.finish()
        
        
    
def main():
  torch.cuda.empty_cache()
  os.environ["TOKENIZERS_PARALLELISM"] = "false"
  train()

if __name__ == '__main__':
  main()
  
    
    
    
    