import torch
from torch import nn
import torch.optim as optim
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.modeling_outputs import ModelOutput


def get_optim(model, args):
    if args.opt == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.opt == 'adamw':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    return optimizer


class Aggregator(nn.Module):
    def __init__(self, n_features=None, aggregation='avg'):
        super(Aggregator, self).__init__()
        self.aggregation = aggregation
        if self.aggregation == 'avg':
            self.pooler= nn.AdaptiveAvgPool1d(1)
        elif self.aggregation == 'attn':
            self.attn = nn.Sequential(
            nn.Linear(n_features, n_features//2),
            nn.Tanh(),
            nn.Linear(n_features//2, 1)
            )
        elif self.aggregation == 'cls_token':
            pass
        else:
            raise NotImplementedError("Aggregation [{}] is not implemented".format(aggregation))
    
    def forward(self, x):
        if self.aggregation == 'avg':
            x = self.pooler(x.permute(0, 2, 1)).squeeze(-1)
        elif self.aggregation == 'attn':
            A = self.attn(x)
            A = F.softmax(A, dim=-1)
            x = torch.bmm(A, x)
            x = x.squeeze(0)
        elif self.aggregation == 'cls_token':
            x = x[:, 0, :]
        return x


@dataclass
class ModelOutputs(ModelOutput):
    """
    Base class for outputs of image classification models.

    Args:
        features (`torch.FloatTensor` of shape `(batch_size, num_features)`):
            The last feature map (after the encoder).
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
            (also called feature maps) of the model at the output of each stage.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    features: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    moe_features: torch.FloatTensor = None
    moe_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
