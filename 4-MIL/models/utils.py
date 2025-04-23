import torch
from torch import nn


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
    

def get_model(args):
    if args.backbone == 'PathAgents':
        from .PathAgents import PathAgents
        model = PathAgents(d_in=args.feature_dim, d_model=args.d_model, 
                           d_state=args.d_state, n_layers=args.n_layers, 
                           n_views=args.n_views, n_classes=args.n_classes, 
                           dropout=args.dropout, task=args.task)
    return model