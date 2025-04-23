import math
import torch
import torch.nn as nn
from .utils import ModelOutputs
from mamba_ssm import Mamba
from einops import rearrange
import torch.nn.functional as F


class LineaEmbedding(nn.Module):
    def __init__(self, n_regions, embed_dim, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(n_regions, embed_dim)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.input_norm = nn.BatchNorm1d(n_regions)

    def forward(self, x):
        # x: [B, L, C]
        x = x.transpose(1, 2)
        x = self.input_norm(x)
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
    

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=1000):
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


class LinearGather(nn.Module):
    def __init__(self, d_model, n_views):
        super().__init__()
        self.attention = nn.Linear(d_model, n_views)

    def forward(self, x):
        # x: [B, L, V, C]
        B, L, V, C = x.size()

        x_reshaped = x.view(B * L, V, C)
        view_summary = x_reshaped.mean(dim=2)  # [B*L, V]
        scores = self.attention(view_summary)  # [B*L, V]
        scores = scores.view(B, L, V) # [B, L, V]

        attention_weights = F.softmax(scores, dim=-1)  # [B, L, V]
        attention_weights = attention_weights.unsqueeze(-1)  # [B, L, V, 1]

        # Compute weighted sum of views
        gathered = (x * attention_weights).sum(dim=2)  # [B, L, C]

        return gathered


class MultiViewMamba(nn.Module):
    def __init__(self, d_model, d_state, n_views, gather=None, dropout=0.1):
        super().__init__()
        self.models = nn.ModuleList([nn.Sequential(nn.LayerNorm(d_model), 
                                                   Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2),
                                                   nn.Dropout(dropout)) 
                                                   for _ in range(n_views)])
        self.gather = gather

    def forward(self, x):
        # x: [B, L, V, C]
        h = []
        for i, model in enumerate(self.models):
            h.append(model(x[..., i, :]))
        h = torch.stack(h, dim=2)

        if self.gather is not None:
            h = self.gather(h) # [B, L, C]
        else:
            h += torch.mean(h, dim=2, keepdim=True) # [B, L, V, C]

        return h


class PathAgents(nn.Module):
    def __init__(self, d_in, d_model=512, d_state=4, n_layers=2,
                 n_views=8, n_classes=4, dropout=0.1, task='grading'):
        super().__init__()
        self.n_views = n_views
        self.d_model = d_model
        self.d_state = d_state
        self.n_classes = n_classes

        self.embedding = LineaEmbedding(d_in, d_model, dropout)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model)
        self.encoder = []
        for _ in range(n_layers - 1):
            self.encoder.append(MultiViewMamba(d_model, d_state, n_views, gather=None, dropout=dropout))
        self.encoder.append(MultiViewMamba(d_model, d_state, n_views, 
                                           gather=LinearGather(d_model, n_views),
                                           dropout=dropout))
        self.encoder = nn.ModuleList(self.encoder)
        self.pooler = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, n_classes)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.task = task
        assert task in ['grading', 'subtyping', 'survival'], \
            f"task must be one of ['grading', 'subtyping', 'survival'], but got {task}"

        # self.contrast_head = nn.Sequential(
        #     nn.Linear(self.d_model, self.d_model),
        #     nn.ReLU(),
        #     nn.Linear(self.d_model, 128)
        # )

        self._init_params()

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
        x = self.positional_encoding(x)
        x = rearrange(x, '(b v) l c -> b l v c', b=B, v=V)

        for layer in self.encoder:
            x = layer(x)
        # x: [B, L, C]
        x = self.pooler(x.transpose(1, 2)).squeeze(-1)  # [B, C]
        features = self.norm(x)
        logits = self.classifier(x)

        return features, logits

    def cls_forward(self, x):
        features, logits = self.feature_forward(x)
        y_hat = torch.argmax(logits, dim=1)
        y_prob = F.softmax(logits, dim=1)
        return ModelOutputs(features=features, logits=logits, y_hat=y_hat, y_prob=y_prob)
    
    def surv_forward(self, x):
        features, logits = self.feature_forward(x)
        y_hat = torch.argmax(logits, dim=1)
        hazards = torch.sigmoid(logits)
        surv = torch.cumprod(1 - hazards, dim=1)
        return ModelOutputs(features=features, logits=logits, hazards=hazards, surv=surv, y_hat=y_hat)
    
    def forward(self, x):
        if self.task == 'grading' or self.task == 'subtyping':
            return self.cls_forward(x)
        elif self.task == 'survival':
            return self.surv_forward(x)
        else:
            raise NotImplementedError
    