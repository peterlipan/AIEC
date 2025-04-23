import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import ModelOutputs


class Pooler(nn.Module):
    def __init__(self, d_in, d_model, n_classes, 
                 p_method='mean', activation='tanh', 
                 surv_classes=4, task='cls'):
        super(Pooler, self).__init__()
        self.p_method = p_method
        self.d_model = d_model
        self.linear = nn.Linear(d_in, d_model)
        self.classifier = nn.Linear(d_model, n_classes) # if uni survival task, n_classes = survival_classes
        self.task = task

        if 'cls' == task and 'surv' == task:
            self.hazard_layer = nn.Linear(d_model, surv_classes)

        if p_method == 'mean':
            self.pooler = nn.AdaptiveAvgPool1d(1)
        elif p_method == 'max':
            self.pooler = nn.AdaptiveMaxPool1d(1)
        else:
            raise NotImplementedError
        
        if activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            raise NotImplementedError
    
    def encode(self, data):
        # the shared forward process for both classification and survival tasks
        # return the features (B, C) before the last linear layer
        x = data['x']
        x = self.pooler(x.transpose(1, 2)).squeeze(-1)
        x = self.linear(x)
        features = self.act(x)
        return features
           
    def cls_forward(self, data):
        features = self.encode(data)
        logits = self.classifier(features)
        y_hat = torch.argmax(logits, dim=1)
        y_prob = F.softmax(logits, dim=1)

        return ModelOutputs(features=features, logits=logits, y_hat=y_hat, y_prob=y_prob)
    
    def surv_forward(self, data):
        features = self.encode(data)
        logits = self.classifier(features)

        y_hat = torch.argmax(logits, dim=1)
        hazards = torch.sigmoid(logits)
        surv = torch.cumprod(1 - hazards, dim=1)
        return ModelOutputs(features=features, logits=logits, hazards=hazards, surv=surv, y_hat=y_hat)

    def multitask_forward(self, data):
        features = self.encode(data)
        cls_logits = self.classifier(features)
        y_hat = torch.argmax(cls_logits, dim=1)
        y_prob = F.softmax(cls_logits, dim=1)

        # survival task
        surv_logits = self.hazard_layer(features)
        surv_y_hat = torch.argmax(surv_logits, dim=1)
        hazards = torch.sigmoid(surv_logits)
        surv = torch.cumprod(1 - hazards, dim=1)

        return ModelOutputs(features=features, cls_logits=cls_logits, y_hat=y_hat, y_prob=y_prob, 
                            surv_logits=surv_logits, surv_y_hat=surv_y_hat, hazards=hazards, surv=surv)

    def forward(self, data):
        if 'cls' == self.task and 'surv' == self.task:
            return self.multitask_forward(data)
        elif 'cls' == self.task:
            return self.cls_forward(data)
        elif 'surv' == self.task:
            return self.surv_forward(data)
        else:
            raise NotImplementedError
        