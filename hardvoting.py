import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import json
import torch
import torch.nn.functional as F

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

# 폴더에 있는 파일 불러오기
path_dir = "./hardvoting/"
file_list = [path_dir + file for file in os.listdir(path_dir) if file.endswith(".csv")]

# 모든 file을 dataframe으로 저장
df_list = []
for file in file_list:
    df = pd.read_csv(file)
    df_list.append(df)

# 새로운 df 만들기
new_df = pd.DataFrame([], columns=["id", "pred_label", "probs"])

# hard voting
for i in tqdm(range(7765)):  # 7765
    df_dict = {}
    prob = np.array([0.0] * 30)

    def return_dict(d):
        return df_dict[d]

    for df in df_list:
        # label count
        try:
            df_dict[df["pred_label"][i]] += 1
        except:
            df_dict[df["pred_label"][i]] = 1

        # prob
        prob += np.array(json.loads(df["probs"][i]))

    # prob
    prob /= len(df_list)
    prob = list(prob)

    # max 값을 갖는 key 찾기
    # 값이 같은게 없다면 --> 개수 많은 것 선택
    if len(df_dict.values()) == len(set(df_dict.values())):
        key_max = max(df_dict.keys(), key=return_dict)
    # 같은 값이 있다면 --> probs 높은 것 선택
    else:
        max_idx = prob.index(max(prob))
        key_max = label_list[max_idx]

    new_df.loc[len(new_df)] = [i, key_max, prob]


# csv 저장
new_df.to_csv("ws2.csv", index=False)
