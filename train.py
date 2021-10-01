import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import glob
import re
import numpy as np
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from load_data import *
import random
from sklearn.model_selection import StratifiedKFold
import argparse
from model import REmodel
import wandb
from focal_loss import FocalLoss
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from MyTrainer import *


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def increment_path(path, exist_ok=False):
    """Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.
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
    label_list = [
        "no_relation",
        "org:top_members/employees",
        "org:members",
        "org:product",
        "per:title",
        "org:alternate_names",
        "per:employee_of",
        "org:place_of_headquarters",
        "per:product",
        "org:number_of_employees/members",
        "per:children",
        "per:place_of_residence",
        "per:alternate_names",
        "per:other_family",
        "per:colleagues",
        "per:origin",
        "per:siblings",
        "per:spouse",
        "org:founded",
        "org:political/religious_affiliation",
        "org:member_of",
        "per:parents",
        "org:dissolved",
        "per:schools_attended",
        "per:date_of_death",
        "per:date_of_birth",
        "per:place_of_birth",
        "per:place_of_death",
        "org:founded_by",
        "per:religion",
    ]
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
    """validation을 위한 metrics function"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

    return {
        "micro f1 score": f1,
        "auprc": auprc,
        "accuracy": acc,
    }


def label_to_num(label):
    num_label = []
    with open("dict_label_to_num.pkl", "rb") as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def train():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # set random seed
    seed_everything(args.seed)

    # load model and tokenizer
    MODEL_NAME = "klue/roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load data
    default_dataset = load_data("../dataset/train/train_revised.csv")
    default_label = label_to_num(default_dataset["label"].values)

    # class weight
    if args.class_weight == True:
        class_weight = Counter(default_label)
        class_weight = list(class_weight.values())
        for i in range(len(class_weight)):
            class_weight[i] = sum(class_weight) / (len(class_weight) * class_weight[i])
            class_weight = torch.tensor(class_weight).to(device=device, dtype=torch.half)
    else:
        class_weight = None
    # K-fold
    kfold = StratifiedKFold(n_splits=args.fold, shuffle=True, random_state=args.seed)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(default_dataset, default_label)):
        print(f"{fold} FOLD")
        run = wandb.init(
            project="klue",
            entity="quarter100",
            name="kfold" + "fold" + str(fold),
        )

        train_dataset = default_dataset.iloc[train_idx]
        valid_dataset = default_dataset.iloc[val_idx]
 
        train_label = label_to_num(train_dataset["label"].values)
        valid_label = label_to_num(valid_dataset["label"].values)

        train_label2 = label_to_num(default_dataset['label'].iloc[train_idx].values)
        valid_label2 = label_to_num(default_dataset['label'].iloc[val_idx].values)

        # tokenizing dataset
        tokenized_train = tokenized_dataset(train_dataset, tokenizer)
        tokenized_valid = tokenized_dataset(valid_dataset, tokenizer)

        # make dataset for pytorch.
        RE_train_dataset = RE_Dataset(tokenized_train, train_label)
        RE_valid_dataset = RE_Dataset(tokenized_valid, valid_label)
        #print(len(RE_train_dataset),len(RE_valid_dataset))
        #print(len(RE_train_dataset2),len(RE_valid_dataset2))

        # setting model hyperparameter
        model = REmodel(MODEL_NAME, device)
        model.to(device)
        model.model.resize_token_embeddings(tokenizer.vocab_size + 16)

        save_dir = increment_path("./results/" + MODEL_NAME)

        # 사용한 option 외에도 다양한 option들이 있습니다.
        # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
        training_args = TrainingArguments(
            output_dir=save_dir,  # output directory
            save_total_limit=1,  # number of total save model.
            save_steps=args.save_steps,  # model saving step.
            num_train_epochs=args.epochs,  # total number of training epochs
            learning_rate=args.lr,  # learning_rate
            per_device_train_batch_size=args.batch,  # batch size per device during training
            per_device_eval_batch_size=args.batch_valid,  # batch size for evaluation
            warmup_steps=args.warmup,  # number of warmup steps for learning rate scheduler
            weight_decay=args.weight_decay,  # strength of weight decay
            logging_dir="./logs",  # directory for storing logs
            logging_steps=args.logging_steps,  # log saving step.
            evaluation_strategy="steps",  # evaluation strategy to adopt during training
            eval_steps=args.eval_steps,  # evaluation step.
            load_best_model_at_end=True,
            seed=args.seed,
            metric_for_best_model="micro f1 score",
            label_smoothing_factor=args.label_smoothing_factor,
            report_to="wandb",
            dataloader_num_workers=args.dataloader_num_workers,
            # fp16=True,
            # group_by_length=True,
        )
        trainer = MyTrainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=RE_train_dataset,  # training dataset
            eval_dataset=RE_valid_dataset,  # evaluation dataset
            compute_metrics=compute_metrics,  # define metrics function
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            loss_name=args.loss,
            class_weight=class_weight,
        )

        # train model
        trainer.train()
        model_object_file_path = "./best_model/" + "fold" + str(fold) + "/pytorch_model.bin"
        if not os.path.exists(model_object_file_path):
            os.makedirs(model_object_file_path)
        torch.save(model.state_dict(), model_object_file_path)
        run.finish()


def main():
    torch.cuda.empty_cache()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--fold", type=int, default=5, help="fold (default: 5)")
    parser.add_argument("--model", type=str, default="klue/roberta-large", help="model type (default: klue/roberta-large)")
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs to train (default: 5)")
    parser.add_argument("--loss", type=str, default="FocalLoss", help="train loss (default: LabelSmoothLoss)")
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate (default: 3e-5)")
    parser.add_argument("--batch", type=int, default=32, help="input batch size for training (default: 32)")
    parser.add_argument("--batch_valid", type=int, default=32, help="input batch size for validing (default: 32)")
    parser.add_argument("--warmup", type=int, default=406, help="warmup_steps (default: 406)")
    parser.add_argument("--eval_steps", type=int, default=250, help="eval_steps (default: 250)")
    parser.add_argument("--save_steps", type=int, default=250, help="save_steps (default: 250)")
    parser.add_argument("--logging_steps", type=int, default=50, help="logging_steps (default: 100)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight_decay (default: 0.01)")
    parser.add_argument("--dataloader_num_workers", type=int, default=2, help="dataloader_num (default: 2)")
    parser.add_argument("--label_smoothing_factor", type=float, default=0.1, help="label_smothing_factor (default: 0.1)")
    parser.add_argument("--class_weight", type=bool, default=False, help="class_weight (default: false)")

    args = parser.parse_args()
    main()
