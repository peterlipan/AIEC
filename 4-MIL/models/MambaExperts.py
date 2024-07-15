import torch
import torch.nn as nn
from transformers import MambaConfig
from .utils import Aggregator, ModelOutputs
from mamba_ssm import Mamba, Mamba2
from .pretrained_mamba import MyMamba, OfficialMamba


class MambaExperts(nn.Module):
    def __init__(self, d_in=1024, d_model=512, d_state=64, n_experts=8, n_classes=2, dropout=0.1, layers=2, act='relu', pretrained='', aggregation='avg'):
        super(MambaExperts, self).__init__()

        if pretrained:
            self.config = MambaConfig.from_pretrained(pretrained)
            self.config.num_hidden_layers = layers
            self.layers = layers
            self.d_model = self.config.hidden_size
        else:
            self.d_model = d_model
            self.d_state = d_state
            self.layers = layers

        self._fc1 = [nn.Linear(d_in, d_model, bias=False)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]

        self._fc1 = nn.Sequential(*self._fc1)
        
        self.n_experts = n_experts
        self.norm = nn.LayerNorm(d_model)
        self.experts = nn.ModuleList()
        self.pretrained = pretrained

        self.aggregation = aggregation

        for _ in range(n_experts):
            temp = MyMamba(config=self.confg) if pretrained else OfficialMamba(d_model=self.d_model, d_state=self.d_state, layers=self.layers, mamba2=False)
            self.experts.append(temp)
            
        self.classifier = nn.Linear(self.d_model, n_classes)
        self.aggregate = Aggregator(aggregation)

    def forward(self, x):
        # x: list, [n_views, 1, L, C]
        # features: [B, n_views, d_model]
        # logits: [B, n_views, n_classes]
        features = []
        logits = []
        x = [self._fc1(view) for view in x]
        for i, expert in enumerate(self.experts):
            exp = self.norm(expert(x[i]))
            exp = self.aggregate(exp)
            pred = self.classifier(exp)
            features.append(exp)
            logits.append(pred)
        features = torch.stack(features, dim=1)
        logits = torch.stack(logits, dim=1)

        moe_features = features.mean(dim=1)
        moe_logits = self.classifier(moe_features)

        return ModelOutputs(features=features, logits=logits, moe_features=moe_features, moe_logits=moe_logits)
    