import torch
from torch import nn
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.modeling_outputs import ModelOutput


class Aggregator(nn.Module):
    def __init__(self, n_features=None, aggregation='avg'):
        super(Aggregator, self).__init__()
        self.aggregation = aggregation
        if self.aggregation == 'avg':
            self.pooler= nn.AdaptiveAvgPool1d(1)
        elif self.aggregation == 'attn':
            self.attn = nn.Sequential(
            nn.Linear(n_features, n_features//2),
            nn.Tanh(),
            nn.Linear(n_features//2, 1)
            )
        elif self.aggregation == 'cls_token':
            pass
        else:
            raise NotImplementedError("Aggregation [{}] is not implemented".format(aggregation))
    
    def forward(self, x):
        if self.aggregation == 'avg':
            x = self.pooler(x.permute(0, 2, 1)).squeeze(-1)
        elif self.aggregation == 'attn':
            A = self.attn(x)
            A = F.softmax(A, dim=-1)
            x = torch.bmm(A, x)
            x = x.squeeze(0)
        elif self.aggregation == 'cls_token':
            x = x[..., -1, :]
        return x


class ModelOutputs:
    def __init__(self, features=None, logits=None, **kwargs):
        self.dict = {'features': features, 'logits': logits}
        self.dict.update(kwargs)
    
    def __getitem__(self, key):
        return self.dict[key]

    def __setitem__(self, key, value):
        self.dict[key] = value
    
    def __str__(self):
        return str(self.dict)

    def __repr__(self):
        return str(self.dict)

    def __getattr__(self, key):
        return self.dict[key]
