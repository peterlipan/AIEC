import torch
from torch import nn
from .gather import GatherLayer

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


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
        device = features.device

        features = F.normalize(features, dim=-1)

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
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class CrossViewConsistency(nn.Module):
    def __init__(self, batch_size, world_size):
        super(CrossViewConsistency, self).__init__()
        self.batch_size = batch_size
        self.world_size = world_size
        self.criteria = SupConLoss()
    
    def forward(self, agent_features, labels):
        # agent_features: [B, n_views, 128]
        N = self.batch_size * self.world_size

        if self.world_size > 1:
            agent_features = torch.cat(GatherLayer.apply(agent_features), dim=0)
            labels = torch.cat(GatherLayer.apply(labels), dim=0)
        
        loss = self.criteria(agent_features, labels)
        
        return loss
        

class KLDivLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, logits1, logits2):
        assert logits1.size() == logits2.size(), f'logits1: {logits1.size()}, logits2: {logits2.size()}'
        softmax1 = self.softmax(logits1)
        softmax2 = self.softmax(logits2)

        probability_loss = self.criterion(softmax1.log(), softmax2)
        return probability_loss


class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, features1, features2):
        loss = nn.functional.mse_loss(features1, features2)
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


def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
   
    S_padded = torch.cat([torch.ones_like(c), S], 1) 

    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards

    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y)+eps) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    ce_l = - c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps))
    loss = (1-alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss


class CrossEntropySurvLoss(nn.Module):
    def __init__(self, alpha=0.15):
        super().__init__()
        self.alpha = alpha

    def forward(self, outputs, data): 

        return ce_loss(outputs.hazards, outputs.surv, data['surv_label'], data['c'], alpha=self.alpha)


class CrossEntropyClsLoss(nn.Module):
    def forward(self, outputs, data):
        return F.cross_entropy(
            outputs['logits'],
            data['label'],
            # label_smoothing=0.1,
        )



class NLLSurvLoss(nn.Module):
    def __init__(self, alpha=0.15):
        super().__init__()
        self.alpha = alpha

    def forward(self, outputs, data):

        return nll_loss(outputs.hazards, outputs.surv, data['surv_label'], data['c'], alpha=self.alpha)


class CoxSurvLoss(nn.Module):
    def forward(self, outputs, data):
        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
        hazards, S, c = outputs.hazards, outputs.surv, data['c']
        device = hazards.device
        current_batch_len = len(S)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i,j] = S[j] >= S[i]

        R_mat = torch.FloatTensor(R_mat).to(device)
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * (1-c))
        return loss_cox