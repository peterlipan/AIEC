import torch
import torch.nn as nn
from utils import Aggregator, ModelOutputs
from mamba_ssm.modules.mamba_simple import Mamba


class SelfAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio

        self.layer_norm = nn.LayerNorm(in_channels)
        self.query_conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels // reduction_ratio, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels // reduction_ratio, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: Input tensor with shape [B, L, C]
        """
        x = self.layer_norm(x)
        x = x.permute(0, 2, 1)

        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)

        energy = torch.bmm(query, key.permute(0, 2, 1))
        attention = self.softmax(energy)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.permute(0, 2, 1)
        out = self.gamma * out + x

        return out


class MambaExperts(nn.Module):
    def __init__(self, d_in=1024, d_model=512, n_experts=8, n_classes=2, dropout=0.1, layers=2, act='gelu', aggregation='avg', prep='linear'):
        super(MambaExperts, self).__init__()

        if prep == 'linear':
            self._fc1 = [nn.LayerNorm(d_in), nn.Linear(d_in, d_model)]
        elif prep == 'attn':
            self._fc1 = [SelfAttention(d_in)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]

        self._fc1 = nn.Sequential(*self._fc1)
        self.n_experts = n_experts
        self.layers = layers
        self.experts = nn.ModuleList()
        if aggregation == 'cls_token':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        for _ in range(n_experts):
            expert = self.single_expert()
            self.experts.append(expert)
            
        self.classifier = nn.Linear(d_model, n_classes)
        self.aggregate = Aggregator(aggregation)
        self.apply(initialize_weights)
    
    @staticmethod
    def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def single_epxpert(self):
        temp = nn.Sequential()
        for _ in range(self.layers):
            temp.append(
                nn.Sequential(
                    Mamba(
                        d_model=d_model,
                        d_state=16,  
                        d_conv=4,    
                        expand=2,
                    ),
                    nn.LayerNorm(d_model),
                )
            )
        return temp


    def forward(self, x):
        # x: [B, n_views, L, C]
        device = x.device
        # features: [B, n_views, d_model]
        # logits: [B, n_views, n_classes]
        features = torch.Tensor().to(device)
        logits = torch.Tensor().to(device)
        x = self._fc1(x)
        for i, expert in enumerate(self.experts):
            exp = expert(x[:, i, :, :])
            exp = self.aggregate(exp)
            pred = self.classifier(exp)
            features = torch.stack((features, exp), dim=1)
            logits = torch.stack((logits, pred), dim=1)

        return ModelOutputs(features=features, logits=logits)
    