# https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py
# https://arxiv.org/pdf/1802.04712.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from .utils import ModelOutputs

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class DAttention(nn.Module):
    def __init__(self, d_in, n_classes, dropout=True, act='gelu', surv_classes=4, task='cls'):
        super(DAttention, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
        self.feature = [nn.Linear(d_in, 512)]
        
        if act.lower() == 'gelu':
            self.feature += [nn.GELU()]
        else:
            self.feature += [nn.ReLU()]

        if dropout:
            self.feature += [nn.Dropout(0.25)]

        self.feature = nn.Sequential(*self.feature)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, n_classes),
        )

        if 'cls' in task and 'surv' in task:
            self.hazard_layer = nn.Linear(self.L*self.K, surv_classes)
        self.task = task


        # self.apply(initialize_weights)
    
    def encode(self, data):
        # the shared forward process for both classification and survival tasks
        # return the features (B, C) before the last linear layer
        x = data['x']
        feature = self.feature(x)
        feature = feature.squeeze()
        A = self.attention(feature)
        A = torch.transpose(A, -1, -2)  # KxN
        A_raw = A
        A = F.softmax(A, dim=-1)  # softmax over N
        features = torch.mm(A, feature)  # KxL
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
        if 'cls' in self.task and 'surv' in self.task:
            return self.multitask_forward(data)
        elif 'cls' in self.task:
            return self.cls_forward(data)
        elif 'surv' in self.task:
            return self.surv_forward(data)
        else:
            raise NotImplementedError


class GatedAttention(nn.Module):
    def __init__(self, d_in, n_classes, dropout=True, act='gelu', surv_classes=4, task='cls'):
        super(GatedAttention, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
        self.feature = [nn.Linear(d_in, 512)]
        if act.lower() == 'gelu':
            self.feature += [nn.GELU()]
        else:
            self.feature += [nn.ReLU()]

        if dropout:
            self.feature += [nn.Dropout(0.25)]

        self.feature = nn.Sequential(*self.feature)

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, n_classes),
        )
        if 'cls' in task and 'surv' in task:
            self.hazard_layer = nn.Linear(self.L*self.K, surv_classes)
        self.task = task
    
    def encode(self, data):
        # the shared forward process for both classification and survival tasks
        # return the features (B, C) before the last linear layer
        x = data['x']
        feature = self.feature(x)
        feature = feature.squeeze()

        A_V = self.attention_V(feature)  # NxD
        A_U = self.attention_U(feature)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        features = torch.mm(A, feature)  # KxL
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
        if 'cls' in self.task and 'surv' in self.task:
            return self.multitask_forward(data)
        elif 'cls' in self.task:
            return self.cls_forward(data)
        elif 'surv' in self.task:
            return self.surv_forward(data)
        else:
            raise NotImplementedError



if __name__ == '__main__':
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    model = DAttention(1024, 2, dropout=False, act='relu')
