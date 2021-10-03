from torch import nn
from transformers import Trainer
from focal_loss import *


class MyTrainer(Trainer):
    def __init__(self, loss_name, class_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_name = loss_name
        self.class_weight = class_weight

    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs.pop("labels")
        outputs = model(**inputs)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logits = outputs.get('logits')

        if self.loss_name == "CrossEntropy":
            custom_loss = torch.nn.CrossEntropyLoss().to(device)
            loss = custom_loss(logits, labels)

        elif self.loss_name == "FocalLoss":
            custom_loss = FocalLoss(gamma=0.5).to(device)
            loss = custom_loss(logits, labels)
            

        elif self.loss_name == "LabelSmoothLoss" and self.label_smoother is not None:
            custom_loss = self.label_smoother(outputs, labels)
            loss = custom_loss.to(device)

        else:  # 위 리스트에 없는 loss_name을 받은 경우 기본 loss를 사용 : crossentropy
            custom_loss = torch.nn.CrossEntropyLoss().to(device)
            loss = custom_loss(outputs[0], labels)

        return (loss, outputs) if return_outputs else loss
