import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import random
import argparse
import glob
import re
import numpy as np
import wandb
from pathlib import Path
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM, Trainer, TrainingArguments, LineByLineTextDataset
import transformers
from load_data import *
from focal_loss import *
from MyTrainer import *
from datetime import date
import uuid

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
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
    #return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices, sample_weight = [0]+[1]*29) * 100.0

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
    seed_everything(args.seed)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # load model and tokenizer
    MODEL_NAME = args.model
    print(f'MODEL_NAME={MODEL_NAME}')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path="en_to_kor_final.txt",
        block_size=512,
    )

    date_time = date.today().isoformat()
    
    # setting model hyperparameter
    model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30

    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, config=model_config)
    model.resize_token_embeddings(tokenizer.vocab_size + 4)
    model.to(device)

    
    save_dir = increment_path(f'./pretrained_model/{date_time}/{args.model}')
    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir=save_dir,          # output directory
        save_total_limit=3,              # number of total save model.
        save_steps=args.save_steps,                 # model saving step.
        num_train_epochs=args.epochs,              # total number of training epochs
        learning_rate=args.lr,               # learning_rate
        per_device_train_batch_size=args.batch,  # batch size per device during training
        warmup_steps=args.warmup,                # number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,               # strength of weight decay
        seed = args.seed,
        overwrite_output_dir = False,
        fp16=args.fp16,
        fp16_opt_level='O3',
        fp16_full_eval=args.fp16,
        fp16_backend='amp',
        report_to='wandb',
        run_name = f'{args.model}-pretrain',
    )
    print(training_args.parallel_mode)
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        data_collator=data_collator,
        train_dataset=dataset,
    )

    # train model
    #wandb.init(project='P2', group=CFG.MODEL_NAME, name=save_dir.split('/')[-1], tags=CFG.tag, config=CFG)
    
    trainer.train()
    model.save_pretrained('./pretrained/')
    
def main():
    torch.cuda.empty_cache()
    train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--model', type=str, default='klue/roberta-large', help='model type (default: klue/roberta-large)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate (default: 5e-5)')
    parser.add_argument('--batch', type=int, default=20, help='input batch size for training (default: 16)')
    parser.add_argument('--batch_valid', type=int, default=20, help='input batch size for validing (default: 16)')
    parser.add_argument('--warmup', type=int, default=812, help='warmup_steps (default: 200)')
    parser.add_argument('--eval_steps', type=int, default=406, help='eval_steps (default: 406)')
    parser.add_argument('--save_steps', type=int, default=406, help='save_steps (default: 406)')
    parser.add_argument('--logging_steps', type=int, default=100, help='logging_steps (default: 100)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight_decay (default: 0.01)')
    parser.add_argument('--fp16', default=True, action='store_true', help='using fp16 mixed precision')

    args = parser.parse_args()
    import time
    prev = time.time()
    print(args)
    
    '''
    CFG = wandb.config
    CFG.name = 'Baseline'
    CFG.tag = ['Baseline']
    CFG.NUM_FOLD = args.fold
    CFG.FOLD = range(CFG.NUM_FOLD)   # if you want to do just simple test, set this [0]
    CFG.MODEL_NAME = args.model
    '''
    
    main()
    print(f'elapsed time = {(time.time() - prev) / 60} minutes')
