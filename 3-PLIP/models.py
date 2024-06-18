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
        self.encoder = getattr(torchvision.models, backbone)(pretrained=True)

        if backbone.startswith('resnet'):
            n_features = self.encoder.fc.in_features
            self.encoder.fc = Identity()
        elif backbone.startswith('densenet'):
            n_features = self.encoder.classifier.in_features
            self.encoder.classifier = Identity()
        elif backbone.startswith('efficientnet'):
            n_features = self.encoder.classifier[1].in_features
            self.encoder.classifier = Identity()
        
        self.projector = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.ReLU(),
            nn.Linear(n_features, 128, bias=False),
        )
    def forward(self, x_i, x_j):
        
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j
