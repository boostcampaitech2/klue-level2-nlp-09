from transformers import AutoModel, AutoConfig, AutoTokenizer
import os
import torch
import torch.nn as nn
from transformers.utils.dummy_pt_objects import AutoModelWithLMHead
from torch.cuda.amp import autocast

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
        outputs = self.model(input_ids = input_ids.to(device=self.device), attention_mask=attention_mask.to(device=self.device))[0]
        batch_size= len(start_obj_idx)
        sub_logit=outputs[0,start_sub_idx[0],:].view(1,1,-1)
        obj_logit=outputs[0,start_obj_idx[0],:].view(1,1,-1)
        for i in range(1,batch_size):
            sub_logit=torch.cat((sub_logit,outputs[i,start_sub_idx[i],:].view(1,1,-1)), 0)
            obj_logit=torch.cat((obj_logit,outputs[i,start_obj_idx[i],:].view(1,1,-1)), 0)
        concat_logit = torch.cat((sub_logit, obj_logit),2).to(device=self.device)
        lin_output = self.lin1(concat_logit)
        lin_output = self.relu(lin_output)
        lin_output = self.lin2(lin_output)
        lin_output = lin_output.view(batch_size,-1)
        return {"logits": lin_output}