import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForSequenceClassification


class Model(nn.Module):
    def __init__(self, MODEL_NAME):
        super().__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


        self.model_config= AutoConfig.from_pretrained(MODEL_NAME)
        self.model_config.num_labels= 30
        self.model= AutoModel.from_pretrained(MODEL_NAME, config= self.model_config).to(self.device)
        # self.model= AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config= self.model_config).to(self.device)


        self.hidden_dim= 1024
        self.num_labels= 30
        
        self.lstm= nn.LSTM(input_size= self.hidden_dim, hidden_size= self.hidden_dim, num_layers= 2, dropout= 0.2, batch_first= True, bidirectional= True)
        self.fc= nn.Linear(self.hidden_dim*2, self.num_labels)


    
    def forward(self, input_ids, attention_mask):
        output= self.model(input_ids= input_ids.to(self.device), attention_mask= attention_mask.to(self.device))[0]
        # print(output.shape)
        # print(output)
        lstm_output, (last_hidden, last_cell)= self.lstm(output)
        # print(last_hidden.shape, last_cell.shape)
        # print(last_hidden[:, 0, :].shape)
        # print(torch.cat((last_hidden[0], last_hidden[1]), dim= 1).shape)

        logits= self.fc(torch.cat((last_hidden[0], last_hidden[1]), dim= 1))
        # print(logits.shape)
        # print(logits)

        return {'logits': logits}