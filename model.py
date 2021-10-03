from transformers import AutoModel, AutoConfig, AutoTokenizer
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

# An Improved Baseline for Sentence-level Relation Extraction(Zhou, 2021) 속 모델을 그대로 구현
class REmodel(nn.Module):
    def __init__(self, model_name, device):
        super(REmodel, self).__init__()
        self.model_name=model_name
        self.model_config = AutoConfig.from_pretrained(self.model_name)
        self.hidden_size = 1024
        self.num_labels=30
        self.device = device
        self.model = AutoModel.from_pretrained(self.model_name, config = self.model_config).to(device = self.device)
        self.lin1 = nn.Linear(in_features = self.hidden_size*2, out_features = self.hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.lin2 = nn.Linear(in_features = self.hidden_size, out_features = self.num_labels)
        
    
    
    @autocast()    
    def forward(self, input_ids, attention_mask, start_obj_idx, start_sub_idx):
        outputs = self.model(input_ids = input_ids.to(device=self.device), attention_mask=attention_mask.to(device=self.device))[0] # [Batch_size, Sequence_Length, Hidden_size]
        batch_size= len(start_obj_idx) 
        # Entity marker @, #의 출력값이 아닌 바로 다음 토큰인 *, ^ 의 출력값을 사용
        sub_logit=outputs[0,start_sub_idx[0]+1,:].view(1,1,-1) # [1, 1, Hidden_size] 1번째 배치의 Subject Entity marker * token이 모델을 거쳐 나온 출력값
        obj_logit=outputs[0,start_obj_idx[0]+1,:].view(1,1,-1) # [1, 1, Hidden_size] 1번째 배치의 Object Entity marker ^ token이 모델을 거쳐 나온 출력값
        # 남은 배치들에 대해서도 추출 후 concatenate를 진행하여 출력값들을 모은다.
        for i in range(1,batch_size):
            sub_logit=torch.cat((sub_logit,outputs[i,start_sub_idx[i]+1,:].view(1,1,-1)), 0) #for문이 끝나면 [Batch_size, 1, Hidden_size]
            obj_logit=torch.cat((obj_logit,outputs[i,start_obj_idx[i]+1,:].view(1,1,-1)), 0) #for문이 끝나면 [Batch_size, 1, Hidden_size]
        concat_logit = torch.cat((sub_logit, obj_logit),2).to(device=self.device) #[Batch_size, 1, Hidden_size*2]
        lin_output = self.lin1(concat_logit) #[Batch_size, 1, Hidden_size]
        lin_output = self.relu(lin_output) #[Batch_size, 1, Hidden_size]
        lin_output = self.dropout(lin_output)
        lin_output = self.lin2(lin_output) #[Batch_size, 1, num_labels]
        lin_output = lin_output.view(batch_size,-1) #[Batch_size, num_labels]
        # Huggingface의 AutoModel들의 Output이 특정한 객체로 반환되며 이 값들 중 logit나 loss등은 자동으로 계산되어 딕셔너리로 담긴다.
        # 이는 Huggingface의 Trainer에서도 이 객체를 통해 loss 계산 및 역전파를 진행하기 때문에 적어도 logit을 아래와 같이 딕셔너리 형태로 제공한다.
        return {"logits": lin_output}