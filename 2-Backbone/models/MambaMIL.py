"""
MambaMIL
"""
import torch
import torch.nn as nn
from transformers import MambaConfig
from utils import ModelOutputs, Aggregator
import torch.nn.functional as F
from .pretrained_mamba import MyMamba


class MambaMIL(nn.Module):
    def __init__(self, d_in, n_classes, dropout, d_model=512, act='gelu', aggregation='avg', layers=2, pretrained=''):
        super().__init__()

        if pretrained:
            self.config = MambaConfig.from_pretrained(pretrained)
            self.config.num_hidden_layers = layers
            self.layers = layers
            self.d_model = self.config.d_model
        else:
            self.config = MambaConfig()
            self.config.num_hidden_layers = layers
            self.config.d_model = d_model
            self.layers = layers
            self.d_model = d_model

        self._fc1 = [nn.Linear(d_in, self.d_model)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]

        self._fc1 = nn.Sequential(*self._fc1)

        self.model = MyMamba.from_pretrained(pretrained, config=self.config) if pretrained else MyMamba(config=self.config)
        if aggregation == 'cls_token':
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))

        self.n_classes = n_classes
        self.classifier = nn.Linear(self.d_model, self.n_classes)
        self.aggregate = Aggregator(aggregation)

    def forward(self, x):
        
        h = self._fc1(x)  # [B, n, d_model]

        if hasattr(self, 'cls_token'):
            cls_token = self.cls_token.expand(h.size(0), -1, -1)
            h = torch.cat((cls_token, h), dim=1)

        h = self.model(h)  # [B, n, d_model]
        hidden = self.aggregate(h)
        pred = self.classifier(hidden)  # [B, n_classes]

        return ModelOutputs(features=hidden, logits=pred, hidden_states=h)
