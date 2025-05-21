import math
import torch
import torch.nn as nn
from .utils import ModelOutputs
from mamba_ssm import Mamba
from einops import rearrange
import torch.nn.functional as F


def swiglu(x):
    a, b = x.chunk(2, dim=-1)
    return a * F.silu(b)


class LinearEmbedding(nn.Module):
    def __init__(self, d_in, d_model, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(d_in, d_model)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.input_norm = nn.BatchNorm1d(d_in)

    def forward(self, x):
        # x: [B, L, C]
        x = x.transpose(1, 2)
        x = self.input_norm(x)
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = self.norm(x)
        x = swiglu(x)
        x = self.dropout(x)
        return x
    

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=20000):
        """
        Sinusoidal positional encoding for sequences.

        Args:
            embed_dim (int): The dimensionality of the embedding space.
            max_len (int): The maximum sequence length to support.
        """
        super(SinusoidalPositionalEncoding, self).__init__()
        
        # Create a matrix to hold the positional encodings
        position = torch.arange(0, max_len).unsqueeze(1)  # Shape: [max_len, 1]
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))  # Shape: [embed_dim // 2]

        # Compute the sinusoidal encodings
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices

        # Register as a buffer so it is not updated during training
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: [1, max_len, embed_dim]

    def forward(self, x):
        """
        Add positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embed_dim].

        Returns:
            torch.Tensor: Input tensor with positional encodings added.
        """
        seq_len = x.size(1)  # Get the sequence length from the input
        return x + self.pe[:, :seq_len, :]


class CrossAgentCommunication(nn.Module):
    def __init__(self, d_model, dropout=0.5, n_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
    
    def forward(self, x):
        B, L, V, C = x.shape
        x = x.permute(0, 1, 3, 2).reshape(B * L, C, V).permute(0, 2, 1)  # [B*L, V, C]
        out, _ = self.attn(x, x, x)
        out = out.permute(0, 2, 1).reshape(B, L, C, V).permute(0, 1, 3, 2)
        return out


class AttentionGather(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
            nn.Dropout(dropout)
        )
        
    def forward(self, h):
        # h: [B, L, V, C]
        B, L, V, C = h.shape
        h_flat = h.view(B*L, V, C)  # [B*L, V, C]
        
        # Compute attention scores
        attn_scores = self.attention(h_flat)  # [B*L, V, 1]
        attn_weights = F.softmax(attn_scores.squeeze(-1), dim=-1)  # [B*L, V]
        
        # Weighted combination
        out_flat = torch.einsum('bv,bvc->bc', attn_weights, h_flat)
        return out_flat.view(B, L, C)  # [B, L, C]


class CrossAttentionGather(nn.Module):
    def __init__(self, d_model, dropout=0.5, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, h):
        B, L, V, C = h.shape
        h = h.permute(0, 2, 1, 3).reshape(B*V, L, C)  # [B*V, L, C]
        context = h.mean(dim=0, keepdim=True)  # [1, L, C]
        context = context.repeat(B*V, 1, 1)  # Match batch size with h
        out, _ = self.attn(context, h, h)
        out = out.view(B, V, L, C).mean(dim=1)  # Aggregate over V
        return out
    

class GatedGather(nn.Module):
    def __init__(self, d_model, n_views, dropout=0.1):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.GELU(),
            nn.LayerNorm(d_model*2),
            nn.Linear(d_model*2, n_views),
            nn.Softmax(dim=-1)
        )
        self.feature_proj = nn.Linear(d_model, d_model)
        
    def forward(self, h):
        """
        Inputs: [B, L, V, C]
        Outputs: [B, L, C]
        """
        B, L, V, C = h.shape
        
        # 1. Compute adaptive weights
        context = h.mean(dim=2)  # Global context [B, L, C]
        weights = self.gate_net(context)  # [B, L, V]
        
        # 2. Feature enhancement
        proj_features = self.feature_proj(h)  # [B, L, V, C]
        
        # 3. Gated fusion
        return torch.einsum('blv,blvc->blc', weights, proj_features)


class MaxGather(nn.Module):
    def __init__(self):
        super().__init__()
        self.pooler = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, h):
        """
        Inputs: [B, L, V, C]
        Outputs: [B, C]
        """
        B, L, V, C = h.shape
        h = h.reshape(B*L, V, C)
        h = self.pooler(h.transpose(1, 2)).squeeze(-1)  # [B*L, C]
        h = h.view(B, L, C)
        return h



class MultiViewMamba(nn.Module):
    def __init__(self, d_model, d_state, n_views, gather=None, dropout=0.1, agent_dropout_rate=0.5):
        super().__init__()
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model), 
                Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2),
                nn.Dropout(dropout)
            ) for _ in range(n_views)
        ])
        self.gather = gather
        self.agent_dropout_rate = agent_dropout_rate
        self.n_views = n_views

    def forward(self, x):
        # x: [B, L, V, C]
        B, L, V, C = x.shape
        assert V == self.n_views, f"Expected {self.n_views} views, got {V}"

        h = []

        # --- Agent dropout ---
        if self.training and self.agent_dropout_rate > 0:
            # [B, V] mask where 1 = keep, 0 = drop
            view_mask = torch.bernoulli(
                (1.0 - self.agent_dropout_rate) * torch.ones(B, V, device=x.device)
            )  # [B, V]

            # Ensure at least one view is kept for each sample
            for i in range(B):
                if view_mask[i].sum() == 0:
                    view_mask[i, torch.randint(0, V, (1,))] = 1
        else:
            view_mask = torch.ones(B, V, device=x.device)

        for i, model in enumerate(self.models):
            x_i = x[..., i, :]  # [B, L, C]
            mask_i = view_mask[:, i].view(B, 1, 1)  # [B, 1, 1]
            out_i = model(x_i) * mask_i  # [B, L, C]
            h.append(out_i)

        agents = torch.stack(h, dim=2)  # [B, L, V, C]

        # Normalize if views are dropped (optional)
        if self.training:
            norm_factors = view_mask.sum(dim=1).clamp(min=1).view(B, 1, 1, 1)
            agents = agents * (V / norm_factors)

        if self.gather is not None:
            h = self.gather(agents)  # [B, L, C]
        else:
            h = agents

        return agents, h


class PathAgents(nn.Module):
    def __init__(self, d_in, d_model=512, d_state=4, n_layers=2,
                 n_views=8, n_classes=4, dropout=0.1, task='grading'):
        super().__init__()
        self.n_views = n_views
        self.d_model = d_model
        self.d_state = d_state
        self.n_classes = n_classes

        self.embedding = LinearEmbedding(d_in, d_model, dropout=dropout)
        self.encoder = []
        for _ in range(n_layers - 1):
            self.encoder.append(MultiViewMamba(d_model, d_state, n_views, 
                                               gather=CrossAgentCommunication(d_model, dropout=dropout),
                                               dropout=dropout))
        self.encoder.append(MultiViewMamba(d_model, d_state, n_views, 
                                           gather=MaxGather(),
                                           dropout=dropout))
        self.encoder = nn.ModuleList(self.encoder)
        
        self.pooler = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, n_classes)
        self.agent_classifier = nn.Linear(d_model, n_classes)
        self.agent_deltas = nn.Parameter(torch.zeros(n_views, d_model, n_classes))  # Learnable diffs

        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.task = task
        assert task in ['grading', 'subtyping', 'survival'], \
            f"task must be one of ['grading', 'subtyping', 'survival'], but got {task}"

        self.projector= nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 64)
        )

        # self._init_params()

    def _init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(module.bias, -bound, bound)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def feature_forward(self, x):
        # x: [B, L, V, C]
        B, L, V, C = x.size()
        x = rearrange(x, 'b l v c -> (b v) l c', b=B, v=V)
        x = self.embedding(x)
        x = rearrange(x, '(b v) l c -> b l v c', b=B, v=V)

        for layer in self.encoder:
            agents, x = layer(x)
        # x: [B, L, C]
        x = self.norm(self.pooler(x.transpose(1, 2)).squeeze(-1))  # [B, C]
        logits = self.classifier(x)

        # agents: [B, L, V, C]
        agents = agents.mean(dim=1)  # [B, V, C]
        agents_logits = self.agent_classifier(agents)  # [B, V, n_classes]

        agents = self.projector(agents)

        return x, logits, agents, agents_logits

    def cls_forward(self, x):
        features, logits, agents, agents_logits = self.feature_forward(x)
        y_hat = torch.argmax(logits, dim=1)
        y_prob = F.softmax(logits, dim=1)
        return ModelOutputs(features=features, logits=logits, agents=agents,
                            agents_logits=agents_logits,
                            y_hat=y_hat, y_prob=y_prob)
    
    def surv_forward(self, x):
        features, logits, agents, agents_logits = self.feature_forward(x)
        y_hat = torch.argmax(logits, dim=1)
        hazards = torch.sigmoid(logits)
        surv = torch.cumprod(1 - hazards, dim=1)
        return ModelOutputs(features=features, logits=logits, agents=agents,
                            agents_logits=agents_logits,
                            hazards=hazards, surv=surv, y_hat=y_hat)
    
    def forward(self, x):
        if self.task == 'grading' or self.task == 'subtyping':
            return self.cls_forward(x)
        elif self.task == 'survival':
            return self.surv_forward(x)
        else:
            raise NotImplementedError
    