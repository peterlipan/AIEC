import torch
from torch import nn
from .MambaMIL import MambaMIL
from .ViT import ViTConfig, ViTForImageClassification
from .transformer import TransformerForSequenceClassification
from .MambaExperts import MambaExperts
from .TransMIL import TransMIL

def create_model(args):
    available_archs = ['Mamba', 'BiMamba', 'SRMamba', 'ViT', 'Transformer', 'Experts', 'TransMIL']
    assert args.backbone in available_archs, f"backbone must be one of {available_archs}"
    if 'Mamba' in args.backbone:
        model = MambaMIL(d_in=args.feature_dim, n_classes=args.num_classes, dropout=args.dropout,
                         d_model=args.d_model, act=args.activation, aggregation=args.agg, layers=args.num_layers, pretrained=args.pretrained)
    elif args.backbone == 'ViT':
        config = ViTConfig().from_pretrained(args.pretrained) if args.pretrained else ViTConfig()
        config.hidden_size = args.feature_dim
        config.num_labels = args.num_classes
        config.num_attention_heads = 4
        config.num_hidden_layers = 3
        model = ViTForImageClassification(config)
    elif args.backbone == 'Transformer':
        model = TransformerForSequenceClassification(d_model=args.feature_dim, max_len=50000, ffn_hidden=args.d_model, n_head=args.num_heads, n_layers=args.num_layers, drop_prob=args.dropout, device='cuda', n_classes=args.num_classes, aggregation=args.agg)
    elif args.backbone == 'Experts':
        if args.pretrained:
            print(f'Loading model from {args.pretrained}')
        model = MambaExperts(d_in=args.feature_dim, d_model=args.d_model, d_state=args.d_state, n_experts=args.n_experts, n_classes=args.num_classes, dropout=args.dropout, layers=args.num_layers, act=args.activation, aggregation=args.agg, pretrained=args.pretrained)

    elif args.backbone == 'TransMIL':
        model = TransMIL(d_in=args.feature_dim, n_classes=args.num_classes)
    else:
        model = None
    return model
