## 코드 사용 GUIDE

1. best_model, prediction, result 폴더를 만든다 !
2. Dict_label_to_num, dict_num_to_label.pkl 파일이 있는지 확인한다.

```bash
mkdir best_model
mkdir prediction
mkdir result
```

2. dataset.py의 Tokenized_dataset에서 원하는 query를 설정해준다.
3. get_config 함수에서 argparse setting 확인한 뒤 train.py 실행
4. 싱글 모델일 경우 inference_single, kfold 모델일 경우 inference_fold 실행 / 이 때, 마찬가지로 argparse check 하기 !
5. Inference_fold.py로 submission을 만들었을 경우,  vote.py를 추가로 실행시켜줘야 한다.

