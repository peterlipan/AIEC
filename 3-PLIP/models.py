import timm
import torch
import torchvision
import torch.nn as nn


class TimmCNNEncoder(torch.nn.Module):
    def __init__(self, model_name: str = 'resnet50.tv_in1k',
                 kwargs: dict = {'features_only': True, 'out_indices': (3,), 'pretrained': True, 'num_classes': 0},
                 pool: bool = True):
        super().__init__()
        assert kwargs.get('pretrained', False), 'only pretrained models are supported'
        self.model = timm.create_model(model_name, **kwargs)
        self.model_name = model_name
        if pool:
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = None

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, list):
            assert len(out) == 1
            out = out[0]
        if self.pool:
            out = self.pool(out).squeeze(-1).squeeze(-1)
        return out


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x


class CreateModel(nn.Module):
    def __init__(self,  n_classes=4, ema=False):
        super().__init__()
        self.encoder = TimmCNNEncoder()
        self.classifier = nn.Linear(1024, n_classes)

        if ema:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.classifier.parameters():
                param.requires_grad = True

    def forward(self, x):
        
        features = self.encoder(x)
        logits = self.classifier(features)
        return features, logits
