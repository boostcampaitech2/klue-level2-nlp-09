## 주의사항
저는 best_model에 저장되어 있는 pytorch_model.bin, config.json 모두 사용합니다.

`inference`에서
AutoModelForSequenceClassification.from_pretrained로 모델을 불러오기 때문에 꼭 위 두 파일이 있어야 합니다 !  
따라서 resize_token_embeddings할 필요가 없습니다.
