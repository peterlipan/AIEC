"""
MambaMIL
"""
import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba
from bimamba import BiMamba
from srmamba import SRMamba
import torch.nn.functional as F


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class MambaMIL(nn.Module):
    def __init__(self, in_dim, n_classes, dropout, act='gelu', survival = False, layer=2, rate=10, backbone="SRMamba"):
        super(MambaMIL, self).__init__()
        self._fc1 = [nn.Linear(in_dim, 512)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]

        self._fc1 = nn.Sequential(*self._fc1)
        self.norm = nn.LayerNorm(512)
        self.layers = nn.ModuleList()
        self.survival = survival

        if backbone == "SRMamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        SRMamba(
                            d_model=512,
                            d_state=16,  
                            d_conv=4,    
                            expand=2,
                        ),
                        )
                )
        elif backbone == "Mamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        Mamba(
                            d_model=512,
                            d_state=16,  
                            d_conv=4,    
                            expand=2,
                        ),
                        )
                )
        elif backbone == "BiMamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        BiMamba(
                            d_model=512,
                            d_state=16,  
                            d_conv=4,    
                            expand=2,
                        ),
                        )
                )
        else:
            raise NotImplementedError("Mamba [{}] is not implemented".format(backbone))

        self.n_classes = n_classes
        self.classifier = nn.Linear(512, self.n_classes)
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.rate = rate
        self.backbone = backbone

        self.apply(initialize_weights)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.expand(1, -1, -1)
        h = x.float()  # [B, n, 1024]
        
        h = self._fc1(h)  # [B, n, 256]

        if self.backbone == "SRMamba":
            for layer in self.layers:
                h_ = h
                h = layer[0](h)
                h = layer[1](h, rate=self.rate)
                h = h + h_
        elif self.backbone == "Mamba" or self.backbone == "BiMamba":
            for layer in self.layers:
                h_ = h
                h = layer[0](h)
                h = layer[1](h)
                h = h + h_

        h = self.norm(h)
        A = self.attention(h) # [B, n, K]
        A = torch.transpose(A, 1, 2)
        A = F.softmax(A, dim=-1) # [B, K, n]
        h = torch.bmm(A, h) # [B, K, 512]
        hidden = h.squeeze(0)

        pred = self.classifier(h)  # [B, n_classes]
        if self.survival:
            Y_hat = torch.topk(pred, 1, dim = 1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat
        return hidden, pred, None
    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        self.layers  = self.layers.to(device)
        
        self.attention = self.attention.to(device)
        self.norm = self.norm.to(device)
        self.classifier = self.classifier.to(device)