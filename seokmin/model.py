import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from torch.cuda.amp import autocast


class BackTransPreTrain():
    def __init__(self, path):
        self.parameters = torch.load(path)

    def load_parameters(self, model_name):
        if model_name == 'klue/roberta-large':
            remove_target = []

            for k in self.parameters.keys():
                if 'roberta' in k:
                    remove_target.append(k)

            print(len(remove_target))

            for r in remove_target:
                self.parameters[r[len('roberta') + 1:]] = self.parameters[r]
                del self.parameters[r]

            remove_key = [
                'lm_head.bias',
                'lm_head.dense.weight',
                'lm_head.dense.bias',
                'lm_head.layer_norm.weight',
                'lm_head.layer_norm.bias'
            ]

            self.parameters['pooler.dense.weight'] = self.parameters['lm_head.dense.weight']
            self.parameters['pooler.dense.bias'] = self.parameters['lm_head.dense.bias']

            for k in remove_key:
                 del self.parameters[k]

            return self.parameters

class Model(nn.Module):
    def __init__(self, MODEL_NAME, pretrain_path):
        super().__init__()

        self.model_config= AutoConfig.from_pretrained(MODEL_NAME)
        self.model_config.num_labels= 30
        self.model= AutoModel.from_pretrained(MODEL_NAME, config= self.model_config)
        if pretrain_path != '':
            bpt = BackTransPreTrain(pretrain_path)
            self.model.load_state_dict(bpt.load_parameters(MODEL_NAME))

        self.hidden_dim= self.model_config.hidden_size # roberta hidden dim = 1024

        self.lstm= nn.LSTM(input_size= self.hidden_dim, hidden_size= self.hidden_dim, num_layers= 2, dropout= 0.2,
                            batch_first= True, bidirectional= True)
        self.fc= nn.Linear(self.hidden_dim*2, self.model_config.num_labels)

    @autocast()
    def forward(self, input_ids, attention_mask):
        # BERT output= (16, 244, 1024) (batch, seq_len, hidden_dim)
        output= self.model(input_ids= input_ids, attention_mask= attention_mask)[0]

        # LSTM last hidden, cell state shape : (2, 244, 1024) (num_layer, seq_len, hidden_size)
        hidden, (last_hidden, last_cell)= self.lstm(output)

        # (16, 1024) (batch, hidden_dim)
        cat_hidden= torch.cat((last_hidden[0], last_hidden[1]), dim= 1)
        logits= self.fc(cat_hidden)
        
        return {'logits': logits}
        