import torch
from torch import nn
from .gather import GatherLayer

import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)  # Add epsilon to avoid log(0)

        # compute mean of log-likelihood over positive
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.ones_like(mask_pos_pairs), mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class CrossViewConsistency(nn.Module):
    def __init__(self):
        super(CrossViewConsistency, self).__init__()
    
    def forward(self, features):
        # features or logits: [1, n_views, C]
        device = features.device
        features = features.squeeze(0)
        # features from different views shall be similar
        view_sim = features.mm(features.t())
        norm = torch.norm(features, p=1, dim=1, keepdim=True)
        view_sim = view_sim / norm
        # ground truth similarity matrix
        gt_sim = torch.ones_like(view_sim) / view_sim.size(0)
        gt_sim = gt_sim.to(device)

        # mask out the diagnol
        mask = torch.ones_like(view_sim) - torch.eye(view_sim.size(0)).to(device)
        mask = mask.bool().to(device)

        # compute the loss
        loss = nn.functional.mse_loss(view_sim[mask], gt_sim[mask])
        return loss


class CrossSampleConsistency(nn.Module):
    def __init__(self, batch_size, world_size):
        super(CrossSampleConsistency, self).__init__()
        self.batch_size = batch_size
        self.world_size = world_size
        self.criteria = SupConLoss(temperature=0.5, contrast_mode='all', base_temperature=0.5)

    def forward(self, features, labels):
        # features: [1, n_views, C]
        # labels: [1,]
        N = self.batch_size * self.world_size
        if self.world_size > 1:
            features = torch.cat(GatherLayer.apply(features), dim=0)
            labels = torch.cat(GatherLayer.apply(labels), dim=0)
        loss = self.criteria(features, labels)
        return loss
