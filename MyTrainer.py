from torch import nn
from transformers import Trainer
from focal_loss import *

class MyTrainer(Trainer):
    def __init__(self, loss_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_name= loss_name

    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs.pop("labels")
        outputs = model(**inputs)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if self.loss_name == 'CrossEntropy':
            custom_loss = torch.nn.CrossEntropyLoss().to(device)
            loss = custom_loss(outputs[0], labels)
        elif self.loss_name == 'FocalLoss' :
            custom_loss = FocalLoss(gamma=0.5).to(device)
            loss = custom_loss(outputs[0], labels)
        elif self.loss_name == 'CrossEntropy_w':
            custom_loss = torch.nn.CrossEntropyLoss(weight = torch.tensor([ 0.1135,  0.2526,  2.5770,  2.8482,  0.5147,  0.8199,  0.3029,  0.9057,
         7.7866, 22.5486,  3.5603,  5.6079,  1.0813,  5.6965,  2.0268,  0.8771,
         7.9583,  1.3614,  2.4052, 11.0442,  0.5800,  2.0814, 16.3990, 13.1992,
         2.5893,  0.9578,  6.5201, 27.0583,  6.9828, 11.2743])).to(device)
            loss = custom_loss(outputs[0], labels)

        else : #위 리스트에 없는 loss_name을 받은 경우 기존 loss를 사용
            custom_loss = torch.nn.CrossEntropyLoss().to(device)
            loss = custom_loss(outputs[0], labels)
            #loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0] #keyerror

        return (loss, outputs) if return_outputs else loss 
