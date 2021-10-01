import pickle as pickle
import os
import pandas as pd
import torch
from transformers import AutoTokenizer


class RE_Dataset(torch.utils.data.Dataset):
    """Dataset 구성을 위한 class."""

    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        for idx in range(len(item["input_ids"])):
            if item["input_ids"][idx] in [32000, 32002]:
                item["start_sub_idx"] = idx
            elif item["input_ids"][idx] in [32004, 32006, 32008, 32010, 32012, 32014]:
                item["start_obj_idx"] = idx
        return item

    def __len__(self):
        return len(self.labels)


def preprocessing_dataset(data):
    """처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    sub_entity, sub_type = [], []
    obj_entity, obj_type = [], []
    sub_idx, obj_idx = [], []
    sentence = []

    for i, [x, y, z] in enumerate(zip(data["subject_entity"], data["object_entity"], data["sentence"])):
        sub_typ = x[1:-1].split(":")[-1].split("'")[-2]
        obj_typ = y[1:-1].split(":")[-1].split("'")[-2]

        if sub_typ == "LOC":
            sub_typ = "ORG"

        for idx_i in range(len(x)):
            if x[idx_i : idx_i + 9] == "start_idx":
                sub_start = int(x[idx_i + 12 :].split(",")[0].strip())
            if x[idx_i : idx_i + 7] == "end_idx":
                sub_end = int(x[idx_i + 10 :].split(",")[0].strip())

            if y[idx_i : idx_i + 9] == "start_idx":
                obj_start = int(y[idx_i + 12 :].split(",")[0].strip())
            if y[idx_i : idx_i + 7] == "end_idx":
                obj_end = int(y[idx_i + 10 :].split(",")[0].strip())

        sub_i = [sub_start, sub_end]
        obj_i = [obj_start, obj_end]

        sub_entity.append(z[sub_i[0] : sub_i[1] + 1])
        obj_entity.append(z[obj_i[0] : obj_i[1] + 1])
        sub_type.append(sub_typ)
        sub_idx.append(sub_i)
        obj_type.append(obj_typ)
        obj_idx.append(obj_i)

        if sub_i[0] < obj_i[0]:
            z = z[: sub_i[0]] + "[SUB:" + sub_typ + "]" + z[sub_i[0] : sub_i[1] + 1] + "[\SUB:" + sub_typ + "]" + z[sub_i[1] + 1 :]
            z = z[: obj_i[0] + 19] + "[OBJ:" + obj_typ + "]" + z[obj_i[0] + 19 : obj_i[1] + 20] + "[\OBJ:" + obj_typ + "]" + z[obj_i[1] + 20 :]
        else:
            z = z[: obj_i[0]] + "[OBJ:" + obj_typ + "]" + z[obj_i[0] : obj_i[1] + 1] + "[\OBJ:" + obj_typ + "]" + z[obj_i[1] + 1 :]
            z = z[: sub_i[0] + 19] + "[SUB:" + sub_typ + "]" + z[sub_i[0] + 19 : sub_i[1] + 20] + "[\SUB:" + sub_typ + "]" + z[sub_i[1] + 20 :]

        sentence.append(z)

    df = pd.DataFrame(
        {
            "id": data["id"],
            "sentence": sentence,
            "subject_entity": sub_entity,
            "object_entity": obj_entity,
            "subject_type": sub_type,
            "object_type": obj_type,
            "label": data["label"],
            "subject_idx": sub_idx,
            "object_idx": obj_idx,
        }
    )
    return df


def load_data(dataset_dir):
    """csv 파일을 경로에 맞게 불러 옵니다."""
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset(pd_dataset)

    return dataset


def tokenized_dataset(dataset, tokenizer):
    """tokenizer에 따라 sentence를 tokenizing 합니다."""
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                "[SUB:PER]",
                "[\SUB:PER]",  # 32000,32001
                "[SUB:ORG]",
                "[\SUB:ORG]",  # 32002,32003
                "[OBJ:PER]",
                "[\OBJ:PER]",  # 32004,32005
                "[OBJ:LOC]",
                "[\OBJ:LOC]",  # 32006,32007
                "[OBJ:POH]",
                "[\OBJ:POH]",  # 32008,32009
                "[OBJ:DAT]",
                "[\OBJ:DAT]",  # 32010,32011
                "[OBJ:NOH]",
                "[\OBJ:NOH]",  # 32012,32013
                "[OBJ:ORG]",
                "[\OBJ:ORG]",
            ]
        }
    )  # 32014,32015
    tokenized_sentences = tokenizer(
        list(dataset["sentence"]), return_tensors="pt", padding=True, truncation=True, max_length=256, add_special_tokens=True, return_token_type_ids=False
    )
    return tokenized_sentences
