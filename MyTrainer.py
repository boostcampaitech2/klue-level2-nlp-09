from torch import nn
from transformers import Trainer
from focal_loss import *


class MyTrainer(Trainer):
    def __init__(self, loss_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_name = loss_name

    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs.pop("labels")
        outputs = model(**inputs)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if self.loss_name == 'CrossEntropy':
            custom_loss = torch.nn.CrossEntropyLoss().to(device)
            loss = custom_loss(outputs[0], labels)
        elif self.loss_name == 'FocalLoss':
            custom_loss = FocalLoss(gamma=0.5).to(device)
            loss = custom_loss(outputs[0], labels)
        else:  # 위 리스트에 없는 loss_name을 받은 경우 기존 loss를 사용
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss