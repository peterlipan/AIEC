import torch
from torch import nn
from .MambaMIL import MambaMIL
from .ViT import ViTConfig, ViTForImageClassification


def create_model(args):
    available_archs = ['Mamba', 'BiMamba', 'SRMamba', 'ViT']
    assert args.backbone in available_archs, f"backbone must be one of {available_archs}"
    if 'Mamba' in args.backbone:
        model = MambaMIL(in_dim=args.feature_dim, n_classes=args.num_classes, dropout=args.dropout,
                         d_model=args.d_model,
                         act=args.activation, aggregation=args.agg, layer=args.num_layers, backbone=args.backbone)
    elif args.backbone == 'ViT':
        config = ViTConfig().from_pretrained(args.pretrained) if args.pretrained else ViTConfig()
        config.num_labels = args.num_classes
        model = ViTForImageClassification(config)
    else:
        model = None
    return model
