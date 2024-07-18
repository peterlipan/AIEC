import torch
from torch import nn
from .TransMIL import TransMIL
from .MambaMIL import MambaMIL
from .MambaExperts import MambaExperts
from .ABMIL import DAttention
from .MambaAgents import MambaAgents

def create_model(args):
    available_archs = ['MambaMIL', 'Experts', 'TransMIL', 'ABMIL', 'MambaAgents']
    assert args.backbone in available_archs, f"backbone must be one of {available_archs}"
    if 'MambaMIL' in args.backbone:
        model = MambaMIL(d_in=args.feature_dim, n_classes=args.num_classes, dropout=args.dropout, d_state=args.d_state,
                         d_model=args.d_model, act=args.activation, num_layers=args.num_layers)
    elif args.backbone == 'Experts':
        if args.pretrained:
            print(f'Loading model from {args.pretrained}')
        model = MambaExperts(d_in=args.feature_dim, d_model=args.d_model, d_state=args.d_state, n_experts=args.n_experts, n_classes=args.num_classes, dropout=args.dropout, layers=args.num_layers, act=args.activation, aggregation=args.agg, pretrained=args.pretrained)
    elif args.backbone == 'TransMIL':
        model = TransMIL(d_in=args.feature_dim, n_classes=args.num_classes)
    elif args.backbone == 'ABMIL':
        model = DAttention(d_in=args.feature_dim, n_classes=args.num_classes, dropout=args.dropout, act=args.activation)
    elif args.backbone == 'MambaAgents':
        model = MambaAgents(d_in=args.feature_dim, d_model=args.d_model, d_state=args.d_state, dropout=args.dropout, n_views=args.n_experts, n_layers=args.num_layers, n_classes=args.num_classes)
    else:
        model = None
    return model
