import torch
import torch.nn as nn
from transformers import MambaConfig
from .utils import Aggregator, ModelOutputs
from mamba_ssm import Mamba, Mamba2
from .pretrained_mamba import MyMamba, OfficialMamba
# from .MambaMIL import Mamba
from nystrom_attention import NystromAttention


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


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class MambaAgents(nn.Module):
    def __init__(self, d_in, d_model, d_state, dropout, n_views, n_layers, n_classes):
        super().__init__()

        self.n_layers = n_layers

        self.in_proj = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.ReLU()
        )

        self.agent_layers = nn.ModuleList([MultiViewMamba(d_model, d_state, n_views) for _ in range(n_layers)])
        # self.post_agent = nn.ModuleList([nn.Sequential(nn.LayerNorm(d_model), Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)) for _ in range(n_layers)])
        self.post_agent = nn.Sequential(*[TransLayer(dim=d_model) for _ in range(n_layers)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, n_classes)

        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.ReLU(),
            nn.Linear(d_model, 64, bias=False),
        )

    def forward(self, x):
        # x: [B, seq_len, n_views, d_in]
        # h: [B, seq_len, n_views, d_model]
        h = self.in_proj(x)

        # agent_features: [B, seq_len, n_views, d_model]
        agent_features = None
        
        for i, layer in enumerate(self.agent_layers):
            res = h
            h = layer(h)
            # log the agent features of the last layer
            if i == self.n_layers - 1:
                agent_features = h
            h = h.mean(dim=2, keepdim=True) + res # broadcast
        
        # merge the views
        # h: [B, seq_len, d_model]
        h = h.mean(dim=2)
        h = self.post_agent(h)
        
        # average pooling to get WSI-level features
        features = self.pool(h.permute(0, 2, 1)).squeeze(-1)
        logits = self.classifier(features)

        # project the agent features for contrastive learning
        # agent_features: [B, seq_len, n_views, 64]
        agent_features = self.projector(agent_features)
        # agent_features: [B, seq_len, n_views, 64] -> [B, n_views, 64]
        agent_features = agent_features.mean(dim=1)

        return ModelOutputs(features=features, logits=logits, hidden_states=h, agent_features=agent_features)