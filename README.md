# klue-level2-nlp-09  

## EDA  
### Random swap  
* eda.py 파일 만들고 코드를 복붙합니다.  
* train.py 에서 from eda import.py를 합니다.  
* easy_data_augmentation 함수를 추가합니다.  
* easy_data_augmentation 함수안에서 random_swap함수를 실행합니다.  
* train_dataset을 load하고 preprocess를 하기전에  
augmented_train_dataset = easy_data_augmented(train_dataset,p) 코드를 추가합니다.  
* random delete도 구현완료  
* p는 augmentation이 일어날 확률입니다.  
* TIP : 꼭 random swap을 하고 random delete를 해야합니다.(index 문제)