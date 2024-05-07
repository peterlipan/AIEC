"""
MambaMIL
"""
import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba
from .bimamba import BiMamba
from .srmamba import SRMamba
from utils import ModelOutputs
import torch.nn.functional as F


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class Aggregator(nn.Module):
    def __init__(self, aggregation='avg'):
        super(Aggregator, self).__init__()
        self.aggregation = aggregation
        if self.aggregation == 'avg':
            self.pooler= nn.AdaptiveAvgPool1d(1)
        elif self.aggregation == 'attention':
            self.attn = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
            )
        elif self.aggregation == 'cls_token':
            pass
        else:
            raise NotImplementedError("Aggregation [{}] is not implemented".format(aggregation))
    
    def forward(self, x):
        if self.aggregation == 'avg':
            x = self.pooler(x.permute(0, 2, 1)).squeeze(-1)
        elif self.aggregation == 'attention':
            A = self.attn(x)
            A = F.softmax(A, dim=-1)
            x = torch.bmm(A, x)
            x = x.squeeze(0)
        elif self.aggregation == 'cls_token':
            x = x[:, 0, :]
        return x


class MambaMIL(nn.Module):
    def __init__(self, in_dim, n_classes, dropout, d_model=512, act='gelu', aggregation='avg',
    survival=False, layer=2, rate=5, backbone="SRMamba"):
        super(MambaMIL, self).__init__()
        self._fc1 = [nn.Linear(in_dim, d_model)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]

        self._fc1 = nn.Sequential(*self._fc1)
        self.norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList()
        self.survival = survival
        if aggregation == 'cls_token':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        if backbone == "SRMamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(d_model),
                        SRMamba(
                            d_model=d_model,
                            d_state=16,  
                            d_conv=4,    
                            expand=2,
                        ),
                        )
                )
        elif backbone == "Mamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(d_model),
                        Mamba(
                            d_model=d_model,
                            d_state=16,  
                            d_conv=4,    
                            expand=2,
                        ),
                        )
                )
        elif backbone == "BiMamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(d_model),
                        BiMamba(
                            d_model=d_model,
                            d_state=16,  
                            d_conv=4,    
                            expand=2,
                        ),
                        )
                )
        else:
            raise NotImplementedError("Mamba [{}] is not implemented".format(backbone))

        self.n_classes = n_classes
        self.classifier = nn.Linear(d_model, self.n_classes)
        self.aggregate = Aggregator(aggregation)
        self.rate = rate
        self.backbone = backbone

        self.apply(initialize_weights)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.expand(1, -1, -1)
        h = x.float()  # [B, n, 1024]
        
        h = self._fc1(h)  # [B, n, d_model]

        if hasattr(self, 'cls_token'):
            cls_token = self.cls_token.expand(h.size(0), -1, -1)
            h = torch.cat((cls_token, h), dim=1)

        if self.backbone == "SRMamba":
            for layer in self.layers:
                h_ = h
                h = layer[0](h)
                h = layer[1](h, rate=self.rate)
                h = h + h_
        elif self.backbone == "Mamba" or self.backbone == "BiMamba":
            for layer in self.layers:
                h_ = h
                h = layer[0](h)
                h = layer[1](h)
                h = h + h_

        h = self.norm(h)
        hidden = self.aggregate(h)

        pred = self.classifier(hidden)  # [B, n_classes]
        if self.survival:
            Y_hat = torch.topk(pred, 1, dim = 1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return ModelOutputs(features=hidden, logits=S)
        return ModelOutputs(features=hidden, logits=pred, hidden_states=h)
    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        self.layers  = self.layers.to(device)
        
        self.aggregate = self.aggregate.to(device)
        self.norm = self.norm.to(device)
        self.classifier = self.classifier.to(device)