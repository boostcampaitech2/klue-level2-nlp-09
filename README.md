# klue-level2-nlp-09



### 코드 사용 GUIDE

1. 자신의 코드에 load_data에 있는 load_data, tokenized_dataset 함수를 변경해주세요 !
2. Train.py, inference.py에 doc string으로 추가한 부분이라고 명시한 부분을 추가해주세요 ! 추가한 부분은 아래와 같으며 모델 생성하는 부분에서 추가를해줬습니다. (추가한 special token의 개수만큼 embedding 차원 수를 맞춰주는 내용입니다.)

``` python
  model.resize_token_embeddings(tokenizer.vocab_size + 4)
```

