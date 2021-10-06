from koeda import AEDA
import pandas as pd
from tqdm import tqdm
import random

"""
JVMNotFoundException 발생시 https://github.com/boostcampaitech2/klue-level2-nlp-09/issues/80 참고
"""

SPACE_TOKEN = "\u241F"


def replace_space(text: str) -> str:
    return text.replace(" ", SPACE_TOKEN)


def revert_space(text: list) -> str:
    clean = " ".join("".join(text).replace(SPACE_TOKEN, " ").split()).strip()
    return clean


# 속도 수정된 aeda
class myAEDA(AEDA):
    def _aeda(self, data: str, p: float) -> str:
        if p is None:
            p = self.ratio

        split_words = self.morpheme_analyzer.morphs(replace_space(data))
        words = self.morpheme_analyzer.morphs(data)

        new_words = []
        q = random.randint(1, int(p * len(words) + 1))
        qs_list = [index for index in range(len(split_words)) if split_words[index] != SPACE_TOKEN]
        qs = random.sample(qs_list, q)

        for j, word in enumerate(split_words):
            if j in qs:
                new_words.append(SPACE_TOKEN)
                new_words.append(self.punctuations[random.randint(0, len(self.punctuations) - 1)])
                new_words.append(SPACE_TOKEN)
                new_words.append(word)
            else:
                new_words.append(word)

        augmented_sentences = revert_space(new_words)

        return augmented_sentences


def make_new_text(sentence, subject_entity, object_entity, punc_ratio):
    aeda = myAEDA(morpheme_analyzer="Okt", punc_ratio=punc_ratio, punctuations=[".", ",", "!", "?", ";", ":"])
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


def start_aeda(train_df, train_label):

    # 늘릴 label 설정
    """NOTE
    label 개수 100 미만: 8배
    label 개수 100~200 미만: 4배
    """
    label_x8 = [
        "org:political/religious_affiliation",
        "per:religion",
        "per:schools_attended",
        "org:dissolved",
        "org:number_of_employees/members",
        "per:place_of_death",
    ]
    label_x4 = ["per:place_of_residence", "per:other_family", "per:place_of_birth", "org:founded_by", "per:product"]

    # index reset
    train_df = train_df.reset_index(drop=True)

    # train data 뒷단에 추가할 dataframe, label list를 선언
    new_df = pd.DataFrame(
        [], columns=["id", "sentence", "subject_entity", "object_entity", "subject_type", "object_type", "label", "subject_idx", "object_idx"]
    )
    new_label = []

    for i in tqdm(range(len(train_df)), desc="augmentation..."):
        # class 확인하여 augmentation 필요한 문장인지 확인
        check_class = train_df.iloc[i]["label"]
        if check_class in label_x8:
            check_num = 8
        elif check_class in label_x4:
            check_num = 4
        else:
            continue
        print(check_num)

        # dataframe에서 문장만 찾음
        sentence = train_df.iloc[i]["sentence"]

        # 문장 길이에 따라 aeda punc_ratio 다르게 설정
        if len(sentence) <= 150:
            punc_ratio = 0.2
        elif len(sentence) <= 300:
            punc_ratio = 0.25
        else:
            punc_ratio = 0.3

        # @, # 안에 있는 것을 찾음 ex: #^PER^조지 해리슨# , @*ORG*비틀즈@
        subject_entity = "@" + sentence.split("@")[1] + "@"
        object_entity = "#" + sentence.split("#")[1] + "#"

        # 새로운 문장 생성
        sentence_list = set([sentence])
        while True:
            new_sentence = make_new_text(sentence, subject_entity, object_entity, punc_ratio)
            sentence_list.add(new_sentence)
            # sentence 포함하여 4/8개 이상이 되면
            if len(sentence_list) >= check_num:
                break

        # 새로 생성된 문장과 문장 정보를 dataframe에 추가
        for s in sentence_list:
            append_new_sentence(new_df, train_df, i, s)
            new_label.append(train_label[i])
        sentence_list.remove(sentence)
        
    # train dataframe 뒷단에 새로운 dataframe 합치기
    aug_df = train_df.append(new_df, ignore_index=True)

    # train label list 뒷단에 새로운 label list 합치기
    train_label.extend(new_label)

    return aug_df, train_label