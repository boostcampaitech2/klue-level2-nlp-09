import torch

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