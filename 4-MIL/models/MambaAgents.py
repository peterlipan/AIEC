import torch
import torch.nn as nn
from transformers import MambaConfig
from .utils import Aggregator, ModelOutputs
from mamba_ssm import Mamba, Mamba2
from .pretrained_mamba import MyMamba, OfficialMamba


class MultiViewMamba(nn.Module):
    def __init__(self, d_model, d_state, n_views):
        super().__init__()
        self.models = nn.ModuleList([nn.Sequential(nn.LayerNorm(d_model), Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)) for _ in range(n_views)])
    
    def forward(self, x):
        # x: [B, seq_len, n_views, d_model]
        # h: [B, seq_len, n_views, d_model]
        h = []
        for i, model in enumerate(self.models):
            h.append(model(x[..., i, :]))
        h = torch.stack(h, dim=2)
        return h


class MambaAgents(nn.Module):
    def __init__(self, d_in, d_model, d_state, dropout, n_views, n_layers, n_classes):
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.Linear(d_in, d_model, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.agent_layers = nn.ModuleList([MultiViewMamba(d_model, d_state, n_views) for _ in range(n_layers)])
        self.post_agent = nn.ModuleList([nn.Sequential(nn.LayerNorm(d_model), Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)) for _ in range(n_layers)])
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        # x: [B, seq_len, n_views, d_in]
        # h: [B, seq_len, n_views, d_model]
        h = self.in_proj(x)
        
        for layer in self.agent_layers:
            res = h
            h = layer(h)
            h = h.mean(dim=2, keepdim=True) + res # broadcast?
        
        # merge the views
        # h: [B, seq_len, d_model]
        h = h.mean(dim=2)
        for layer in self.post_agent:
            res = h
            h = layer(h)
            h = h + res
        
        # the last hidden state to represent the seq
        features = h[:, -1, :]
        logits = self.classifier(features)

        return ModelOutputs(features=features, logits=logits, hidden_states=h)
