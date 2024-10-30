import os
import torch
import wandb
import time
import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
from .metrics import compute_avg_metrics
from .losses import CrossSampleConsistency, CrossViewConsistency


def train(dataloaders, model, criteria, optimizer, scheduler, args, logger):

    train_loader, test_loader = dataloaders
    model.train()
    start = time.time()
    xview_criteria = CrossViewConsistency(batch_size=args.batch_size, world_size=args.world_size)
    cur_iter = 0

    for epoch in range(args.epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        for i, (_, img, labels) in enumerate(train_loader):
            img, labels = img.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            outputs = model(img)
            logits, agent_features = outputs.logits, outputs.agent_features            

            # classification loss
            cls_loss = criteria(logits, labels)

            if agent_features is not None:
                xview_loss = args.lambda_xview * xview_criteria(agent_features, labels)
                loss = cls_loss + xview_loss
            
            else:
                loss = cls_loss

            if args.rank == 0:
                train_loss = loss.item()
                cls_loss_value = cls_loss.item()
                xview_loss_value = xview_loss.item() if agent_features is not None else 0
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if dist.is_available() and dist.is_initialized():
                loss = loss.data.clone()
                dist.all_reduce(loss.div_(dist.get_world_size()))
            
            cur_iter += 1
            if args.rank == 0:
                if cur_iter % 50 == 0:
                    cur_lr = optimizer.param_groups[0]['lr']
                    test_acc, test_f1, test_auc, test_ap, test_bac, test_sens, test_spec, test_prec, test_mcc, test_kappa = validate(epoch,
                        test_loader, model)
                    if logger is not None:
                        logger.log({'test': {'Accuracy': test_acc,
                                             'F1 score': test_f1,
                                             'AUC': test_auc,
                                             'AP': test_ap,
                                             'Balanced Accuracy': test_bac,
                                             'Sensitivity': test_sens,
                                             'Specificity': test_spec,
                                             'Precision': test_prec,
                                             'MCC': test_mcc,
                                             'Kappa': test_kappa},
                                    'train': {'loss': train_loss,
                                                'cls_loss': cls_loss_value,
                                                'xview_loss': xview_loss_value,
                                              'learning_rate': cur_lr}}, )

                    print('\rEpoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.6f || Loss: %.4f' % (
                        epoch, args.epochs, i + 1, len(train_loader), time.time() - start,
                        cur_lr, train_loss), end='', flush=True)

    # validate and save the model
    if args.rank == 0:
        test_acc, test_f1, test_auc, test_ap, test_bac, test_sens, test_spec, test_prec, test_mcc, test_kappa = validate(epoch,
            test_loader, model)
        if logger is not None:
            logger.log({'test': {'Accuracy': test_acc,
                                    'F1 score': test_f1,
                                    'AUC': test_auc,
                                    'AP': test_ap,
                                    'Balanced Accuracy': test_bac,
                                    'Sensitivity': test_sens,
                                    'Specificity': test_spec,
                                    'Precision': test_prec,
                                    'MCC': test_mcc,
                                    'Kappa': test_kappa}})
        print(f"\nFold {args.fold}, Test Accuracy: {test_acc}, Test F1: {test_f1}, Test AUC: {test_auc}, "
                f"Test BAC: {test_bac}, Test Sensitivity: {test_sens}, Test Specificity: {test_spec}, "
                f"Test Precision: {test_prec}, Test MCC: {test_mcc}, Test Kappa: {test_kappa}")

        model_path = os.path.join(args.checkpoints, f"fold_{args.fold}_acc_{test_acc}.pth")
        state_dict = model.module.state_dict() if isinstance(model, DataParallel) or isinstance(model,
                                                                                                DDP) else model.state_dict()
        torch.save(state_dict, model_path)


def validate(epoch, dataloader, model):
    training = model.training
    model.eval()

    ground_truth = torch.Tensor().cuda()
    predictions = torch.Tensor().cuda()
    wsi_names = []

    with torch.no_grad():
        for name, img, label in dataloader:
            img, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True).long()
            outputs = model(img)
            logits = outputs.logits
            pred = F.softmax(logits, dim=1)
            ground_truth = torch.cat((ground_truth, label))
            predictions = torch.cat((predictions, pred))    
            wsi_names.extend(name)    

        acc, f1, auc, ap, bac, sens, spec, prec, mcc, kappa = compute_avg_metrics(ground_truth, predictions, avg='macro')
        preds = predictions.argmax(dim=1).cpu().detach().tolist()
        ground_truth = ground_truth.cpu().detach().tolist()
        write_csv(epoch, wsi_names, preds, ground_truth)
    model.train(training)
    return acc, f1, auc, ap, bac, sens, spec, prec, mcc, kappa


def train_experts(dataloaders, model, criteria, optimizer, scheduler, args, logger):
    train_loader, test_loader = dataloaders
    model.train()
    start = time.time()

    xview = CrossViewConsistency(batch_size=args.batch_size, world_size=args.world_size)

    cur_iter = 0
    for epoch in range(args.epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        for i, (_, img, label) in enumerate(train_loader):
            if isinstance(img, list):
                img = [x.cuda(non_blocking=True) for x in img]
            else:
                img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            outputs = model(img)
            features, logits, agent_features = outputs.features, outputs.logits, outputs.agent_features

            # classification loss
            cls_loss = criteria(logits.view(args.n_experts * args.batch_size, -1), label.repeat(args.n_experts))
            xview_loss = xview(agent_features, label)
            loss  = cls_loss + args.lambda_xview * xview_loss

            if args.rank == 0:
                train_loss = loss.item()
                xview_value = xview_loss.item()
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if dist.is_available() and dist.is_initialized():
                loss = loss.data.clone()
                dist.all_reduce(loss.div_(dist.get_world_size()))

            cur_iter += 1
            if args.rank == 0:
                if cur_iter % 50 == 0:
                    cur_lr = optimizer.param_groups[0]['lr']
                    test_performance = valid_experts(epoch, test_loader, model)
                    if logger is not None:
                        logger.log({'test': test_performance,
                                    'train': {'loss': train_loss,
                                              'xview_loss': xview_value,
                                              'learning_rate': cur_lr}}, )

                    print('\rEpoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.6f || Loss: %.4f' % (
                        epoch, args.epochs, i + 1, len(train_loader), time.time() - start,
                        cur_lr, train_loss), end='', flush=True)

    # validate and save the model
    if args.rank == 0:
        test_performance = valid_experts(epoch, test_loader, model)
        if logger is not None:
            logger.log({'test': test_performance})
        print(f"\nFold {args.fold}, Test Accuracy: {test_acc}, Test F1: {test_f1}, Test AUC: {test_auc}, "
                f"Test BAC: {test_bac}, Test Sensitivity: {test_sens}, Test Specificity: {test_spec}, "
                f"Test Precision: {test_prec}, Test MCC: {test_mcc}, Test Kappa: {test_kappa}")

        model_path = os.path.join(args.checkpoints, f"fold_{args.fold}_acc_{test_acc}.pth")
        state_dict = model.module.state_dict() if isinstance(model, DataParallel) or isinstance(model,
                                                                                                DDP) else model.state_dict()
        torch.save(state_dict, model_path)


def valid_experts(epoch, dataloader, model):

    training = model.training
    model.eval()

    ground_truth = torch.Tensor().cuda()
    moe_probs = torch.Tensor().cuda()
    exp_probs = torch.Tensor().cuda()

    wsi_names = []

    return_dict = {}

    with torch.no_grad():
        for name, img, label in dataloader:
            if isinstance(img, list):
                img = [x.cuda(non_blocking=True) for x in img]
            else:
                img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True).long()
            outputs = model(img)
            logits, moe_logits = outputs.logits, outputs.moe_logits
            # logts: [B, n_experts, n_classes]
            exp_prob = F.softmax(logits, dim=-1)
            exp_probs = torch.cat((exp_probs, exp_prob))

            moe_prob = F.softmax(moe_logits, dim=-1)
            moe_probs = torch.cat((moe_probs, moe_prob))
            ground_truth = torch.cat((ground_truth, label))
            wsi_names.extend(name)
            

        for i in range(exp_probs.shape[1]):
            acc, f1, auc, ap, bac, sens, spec, prec, mcc, kappa = compute_avg_metrics(ground_truth, exp_probs[:, i, :], avg='macro')
            return_dict[f'Expert_{i}'] = {'Accuracy': acc,
                                          'F1 score': f1,
                                          'AUC': auc,
                                          'AP': ap,
                                          'Balanced Accuracy': bac,
                                          'Sensitivity': sens,
                                          'Specificity': spec,
                                          'Precision': prec,
                                          'MCC': mcc,
                                          'Kappa': kappa}
        acc, f1, auc, ap, bac, sens, spec, prec, mcc, kappa = compute_avg_metrics(ground_truth, moe_probs, avg='macro')
        return_dict['Overall'] = {'Accuracy': acc,
                                  'F1 score': f1,
                                  'AUC': auc,
                                  'AP': ap,
                                  'Balanced Accuracy': bac,
                                  'Sensitivity': sens,
                                  'Specificity': spec,
                                  'Precision': prec,
                                  'MCC': mcc,
                                  'Kappa': kappa}

        moe_preds = moe_probs.argmax(dim=-1).cpu().detach().tolist()
        ground_truth = ground_truth.cpu().detach().tolist()
        # write_csv(epoch, wsi_names, moe_preds, ground_truth)


    model.train(training)
    return return_dict

def write_csv(epoch, names, preds, labels):
    path = '/mnt/zhen_chen/AIEC/4-MIL/results.csv'
    if not os.path.exists(path):
        df = pd.DataFrame({'WSI': names, 'Label': labels, f'Epoch_{epoch}': preds})
    else:
        df = pd.read_csv(path)
        assert names == df['WSI'].tolist(), 'WSI names do not match'
        assert labels == df['Label'].tolist(), 'Labels do not match'
        df[f'Epoch_{epoch}'] = preds
    df.to_csv(path, index=False)
