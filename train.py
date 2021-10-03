import os
import torch
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, EarlyStoppingCallback
import argparse
import random
import argparse

import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


import wandb
from dataset import *
from model import *


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--train_path", type=str, default="/opt/ml/dataset/train/train.csv", help="train csv path (default: /opt/ml/dataset/train/train.csv")
    parser.add_argument("--tokenize_option", type=str, default="PUN", help="token option ex) SUB, PUN")
    parser.add_argument("--fold", type=int, default=5, help="fold (default: 5)")
    parser.add_argument("--model", type=str, default="klue/roberta-large", help="model type (default: klue/roberta-large)")
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs to train (default: 5)")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate (default: 1e-5)")
    parser.add_argument("--batch", type=int, default=32, help="input batch size for training (default: 32)")
    parser.add_argument("--batch_valid", type=int, default=32, help="input batch size for validing (default: 32)")
    parser.add_argument("--warmup", type=int, default=0.1, help="warmup_ratio (default: 0.1)")
    parser.add_argument("--eval_steps", type=int, default=250, help="eval_steps (default: 250)")
    parser.add_argument("--save_steps", type=int, default=250, help="save_steps (default: 250)")
    parser.add_argument("--logging_steps", type=int, default=50, help="logging_steps (default: 50)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight_decay (default: 0.01)")
    parser.add_argument("--metric_for_best_model", type=str, default="micro f1 score", help="metric_for_best_model (default: micro f1 score")
    parser.add_argument("--save_dir", type=str, default=f"./result/{args.model}_kfold{fold}_lstm_punc_single", help="save dir (default: - )")
    args = parser.parse_args()

    return args


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
        "normal f1": sklearn.metrics.f1_score(labels, preds, average="micro") * 100.0,
        "auprc": auprc,
        "accuracy": acc,
    }


def train(args):

    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    preprocess = Preprocess(args.train_path, args.tokenize_option)

    all_dataset = preprocess.data
    all_label = all_dataset["label"].values

    kfold = StratifiedKFold(n_splits=args.fold, shuffle=True, random_state=args.seed)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_dataset, all_label)):
        run = wandb.init(project="klue", entity="quarter100", name=f"lstm_punc_add_type_single")
        print(f"fold: {fold} start!")

        # load dataset
        train_dataset = all_dataset.iloc[train_idx]
        val_dataset = all_dataset.iloc[val_idx]

        train_label = preprocess.label_to_num(train_dataset["label"].values)
        val_label = preprocess.label_to_num(val_dataset["label"].values)

         # tokenizing dataset
        tokenized_train, token_size = preprocess.tokenized_dataset(train_dataset, tokenizer)
        tokenized_val, _ = preprocess.tokenized_dataset(val_dataset, tokenizer)

         # make dataset for pytorch.
        trainset = Dataset(tokenized_train, train_label)
        valset = Dataset(tokenized_val, val_label)

        model = Model(args.model)
        model.model.resize_token_embeddings(tokenizer.vocab_size + token_size)
        model.to(device)

        training_args = TrainingArguments(
            output_dir=args.save_dir,
            save_total_limit=2,
            save_steps=args.save_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch_valid,
            label_smoothing_factor=0.1,
            warmup_ratio=args.warmup,
            weight_decay=args.weight_decay,
            logging_dir="./logs",
            logging_steps=args.logging_steps,
            metric_for_best_model=args.metric_for_best_model,
            evaluation_strategy="steps",
            group_by_length=True,
            eval_steps=args.eval_steps,
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=trainset,
            eval_dataset=valset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        trainer.train()
        run.finish()


if __name__ == "__main__":

    args = get_config()
    train(args)
