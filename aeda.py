from koeda import AEDA
import pandas as pd
from tqdm import tqdm

"""
JVMNotFoundException 발생시 https://github.com/boostcampaitech2/klue-level2-nlp-09/issues/80 참고
"""


def make_new_text(sentence, subject_entity, object_entity, punc_ratio):
    aeda = AEDA(morpheme_analyzer="Okt", punc_ratio=punc_ratio, punctuations=[".", ",", "!", "?", ";", ":"])
    while True:
        new_sentence = aeda(sentence)
        # subject, object entity가 깨지지 않는 문장만 생성
        if subject_entity in new_sentence and object_entity in new_sentence:
            break

    return new_sentence


def append_new_sentence(new_df, train_df, i, sentence):
    new_df.loc[len(new_df)] = [
        train_df.loc[i]["id"],
        sentence,
        train_df.loc[i]["subject_entity"],
        train_df.loc[i]["object_entity"],
        train_df.loc[i]["subject_type"],
        train_df.loc[i]["object_type"],
        train_df.loc[i]["label"],
        train_df.loc[i]["subject_idx"],
        train_df.loc[i]["object_idx"],
    ]


def start_aeda(train_df, train_label, num_aeda):
    # num_aeda 체크
    if num_aeda == 0 or num_aeda > 2:
        assert (False, "num_aeda must be 1 or 2")

    # index reset
    train_df = train_df.reset_index(drop=True)

    # train data 뒷단에 추가할 dataframe, label list를 선언
    new_df = pd.DataFrame(
        [], columns=["id", "sentence", "subject_entity", "object_entity", "subject_type", "object_type", "label", "subject_idx", "object_idx"]
    )
    new_label = []

    for i in tqdm(range(len(train_df)), desc="augmentation..."):
        # dataframe에서 문장만 찾음
        sentence = train_df.iloc[i]["sentence"]

        # 문장에 따라 aeda punc_ratio 다르게 설정
        if len(sentence) <= 150:
            punc_ratio = 0.3
        elif len(sentence) <= 300:
            punc_ratio = 0.15
        else:
            punc_ratio = 0.05

        # @, # 안에 있는 것을 찾음 ex: #^PER^조지 해리슨# , @*ORG*비틀즈@
        subject_entity = "@" + sentence.split("@")[1] + "@"
        object_entity = "#" + sentence.split("#")[1] + "#"

        # 원본 문장과 같아선 안됨
        while True:
            new_sentence = make_new_text(sentence, subject_entity, object_entity, punc_ratio)
            if new_sentence != sentence:
                break

        # 논문 기준 2번정도가 안전하다고 판단하여 2번 더 추가(3배로 늘림), but aug생성만 40분 걸립니다 ㅠㅠ
        if num_aeda == 2:
            # 2번 문장이 원본, 1번 문장과 같아서는 안됨
            while True:
                new_sentence2 = make_new_text(sentence, subject_entity, object_entity, punc_ratio)
                if new_sentence2 != sentence and new_sentence2 != new_sentence:
                    break

        # 새로 생성된 문장과 문장 정보를 dataframe에 추가 (2번)
        append_new_sentence(new_df, train_df, i, new_sentence)
        if num_aeda == 2:
            append_new_sentence(new_df, train_df, i, new_sentence2)

        for _ in range(num_aeda):
            new_label.append(train_label[i])

    # train dataframe 뒷단에 새로운 dataframe 합치기
    aug_df = train_df.append(new_df, ignore_index=True)

    # train label list 뒷단에 새로운 label list 합치기
    train_label.extend(new_label)

    return aug_df, train_label