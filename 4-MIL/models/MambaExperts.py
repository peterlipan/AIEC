import torch
import torch.nn as nn
from transformers import MambaConfig
from utils import Aggregator, ModelOutputs
from mamba_ssm import Mamba, Mamba2
from .pretrained_mamba import MyMamba


class MambaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MambaRMSNorm is equivalent to T5LayerNorm and LlamaRMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Mamba2Block(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model, d_state, d_conv, expand)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.mamba(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class SelfAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
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
        x: Input tensor with shape [B, n_views, L, C]
        """
        B, n_views, L, C = x.shape

        # Apply layer normalization
        x = self.layer_norm(x)

        # Reshape to merge batch and n_views dimensions for convolution
        x = x.view(B * n_views, L, C).permute(0, 2, 1)  # Shape: [B * n_views, C, L]

        query = self.query_conv(x)  # Shape: [B * n_views, C // reduction_ratio, L]
        key = self.key_conv(x)      # Shape: [B * n_views, C // reduction_ratio, L]
        value = self.value_conv(x).permute(0, 2, 1)  # Shape: [B * n_views, L, C]

        energy = torch.bmm(query.permute(0, 2, 1), key)  # Shape: [B * n_views, L, L]
        attention = self.softmax(energy)  # Shape: [B * n_views, L, L]

        out = torch.bmm(attention, value)  # Shape: [B * n_views, L, C]
        out = out.permute(0, 2, 1)  # Shape: [B * n_views, C, L]

        # Reshape back to [B, n_views, L, C]
        out = out.contiguous().view(B, n_views, L, C)

        # Apply residual connection
        out = self.gamma * out + x.contiguous().view(B, n_views, L, C)

        return out




class MambaExperts(nn.Module):
    def __init__(self, d_in=1024, d_model=512, d_state=64, n_experts=8, n_classes=2, dropout=0.1, layers=2, act='gelu', aggregation='avg', prep='linear', pretrained=''):
        super(MambaExperts, self).__init__()

        if pretrained:
            self.config = MambaConfig.from_pretrained(pretrained)
            self.config.num_hidden_layers = layers
            self.layers = layers
            self.d_model = self.config.d_model
        else:
            self.layers = layers
            self.d_model = d_model
            self.d_state = d_state


        if prep == 'linear':
            self._fc1 = [nn.LayerNorm(d_in), nn.Linear(d_in, self.d_model)]
        elif prep == 'attn':
            self._fc1 = [SelfAttention(d_in), nn.LayerNorm(d_in), nn.Linear(d_in, self.d_model)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]

        self._fc1 = nn.Sequential(*self._fc1)
        self.n_experts = n_experts
        self.experts = nn.ModuleList()
        self.pretrained = pretrained

        self.aggregation = aggregation

        for _ in range(n_experts):
            temp = MyMamba.from_pretrained(self.pretrained, config=self.config) if pretrained else self.single_expert()
            self.experts.append(temp)
            
        self.classifier = nn.Linear(self.d_model, n_classes)
        self.aggregate = Aggregator(aggregation)
        self.apply(self.initialize_weights)
    
    def single_expert(self):
        temp = []
        for _ in range(self.layers):
            temp.append(
                Mamba2Block(
                    d_model=self.d_model,
                    d_state=self.d_state,
                    d_conv=4,
                    expand=2,
                )
            )
        return nn.Sequential(*temp)
    
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

    def forward(self, x):
        # x: [B, n_views, L, C]
        device = x.device
        # features: [B, n_views, d_model]
        # logits: [B, n_views, n_classes]
        features = []
        logits = []
        x = self._fc1(x)
        for i, expert in enumerate(self.experts):
            exp = expert(x[:, i, :, :])
            exp = self.aggregate(exp)
            pred = self.classifier(exp)
            features.append(exp)
            logits.append(pred)
        features = torch.stack(features, dim=1)
        logits = torch.stack(logits, dim=1)

        moe_features = features.mean(dim=1)
        moe_logits = self.classifier(moe_features)

        return ModelOutputs(features=features, logits=logits, moe_features=moe_features, moe_logits=moe_logits)
    