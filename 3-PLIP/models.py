import torchvision
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x


class CreateModel(nn.Module):
    def __init__(self, backbone='densenet121'):
        super().__init__()
        self.model = getattr(torchvision.models, backbone)(pretrained=True)

        if backbone.startswith('resnet'):
