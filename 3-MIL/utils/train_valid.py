import os
import torch
import wandb
import time
import numpy as np
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
    cudnn.benchmark = False
    cudnn.deterministic = True
    train_loader, test_loader = dataloaders
    model.train()
    start = time.time()

    cur_iter = 0
    for epoch in range(args.epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        for i, (img, label) in enumerate(train_loader):
            img, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True).long()
            outputs = model(img)
            features, logits = outputs.features, outputs.logits

            # classification loss
            loss = criteria(logits, label)

            if args.rank == 0:
                train_loss = loss.item()
                
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
                if cur_iter % 50 == 1:
                    cur_lr = optimizer.param_groups[0]['lr']
                    test_acc, test_f1, test_auc, test_ap, test_bac, test_sens, test_spec, test_prec, test_mcc, test_kappa = validate(
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
                                              'learning_rate': cur_lr}}, )

                    print('\rEpoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.6f || Loss: %.4f' % (
                        epoch, args.epochs, i + 1, len(train_loader), time.time() - start,
                        cur_lr, train_loss), end='', flush=True)

    # validate and save the model
    if args.rank == 0:
        test_acc, test_f1, test_auc, test_ap, test_bac, test_sens, test_spec, test_prec, test_mcc, test_kappa = validate(
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


def validate(dataloader, model):
    training = model.training
    model.eval()

    ground_truth = torch.Tensor().cuda()
    predictions = torch.Tensor().cuda()

    with torch.no_grad():
        for img, label in dataloader:
            img, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True).long()
            outputs = model(img)
            logits = outputs.logits
            pred = F.softmax(logits, dim=1)
            ground_truth = torch.cat((ground_truth, label))
            predictions = torch.cat((predictions, pred))

        acc, f1, auc, ap, bac, sens, spec, prec, mcc, kappa = compute_avg_metrics(ground_truth, predictions, avg='micro')
    model.train(training)
    return acc, f1, auc, ap, bac, sens, spec, prec, mcc, kappa


def train_experts(dataloaders, model, criteria, optimizer, scheduler, args, logger):
    train_loader, test_loader = dataloaders
    model.train()
    start = time.time()

    xsample = CrossSampleConsistency(batch_size=args.batch_size, world_size=args.world_size)
    xview = CrossViewConsistency()

    cur_iter = 0
    for epoch in range(args.epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        for i, (img, label) in enumerate(train_loader):
            img, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True).long()
            outputs = model(img)
            features, logits = outputs.features, outputs.logits

            # classification loss
            xsam_feature_loss = xsample(features, label)
            xsam_logits_loss = xsample(logits, label)
            xview_feature_loss = xview(features)
            xview_logits_loss = xview(logits)

            cls_loss = criteria(logits.view(args.n_experts, -1), label.repeat(args.n_experts))
            overall_cls_loss = criteria(logits.mean(1), label)
            # print(xsam_feature_loss.item(), xsam_logits_loss.item(), xview_feature_loss.item(), xview_logits_loss.item(), cls_loss.item(), overall_cls_loss.item())
            loss = cls_loss + overall_cls_loss + args.lambda_xsam * xsam_feature_loss + args.lambda_xsam * xsam_logits_loss + args.lambda_xview * xview_feature_loss + args.lambda_xview * xview_logits_loss


            if args.rank == 0:
                train_loss = loss.item()
                
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
                    test_performance = valid_experts(test_loader, model)
                    if logger is not None:
                        logger.log({'test': test_performance,
                                    'train': {'loss': train_loss,
                                              'xsample_feature_loss': args.lambda_xsam * xsam_feature_loss.item(),
                                              'xsample_logits_loss': args.lambda_xsam * xsam_logits_loss.item(),
                                              'xview_feature_loss': args.lambda_xview * xview_feature_loss.item(),
                                              'xview_logits_loss': args.lambda_xview * xview_logits_loss.item(),
                                              'learning_rate': cur_lr}}, )

                    print('\rEpoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.6f || Loss: %.4f' % (
                        epoch, args.epochs, i + 1, len(train_loader), time.time() - start,
                        cur_lr, train_loss), end='', flush=True)

    # validate and save the model
    if args.rank == 0:
        test_performance = valid_experts(test_loader, model)
        if logger is not None:
            logger.log({'test': test_performance})
        print(f"\nFold {args.fold}, Test Accuracy: {test_acc}, Test F1: {test_f1}, Test AUC: {test_auc}, "
                f"Test BAC: {test_bac}, Test Sensitivity: {test_sens}, Test Specificity: {test_spec}, "
                f"Test Precision: {test_prec}, Test MCC: {test_mcc}, Test Kappa: {test_kappa}")

        model_path = os.path.join(args.checkpoints, f"fold_{args.fold}_acc_{test_acc}.pth")
        state_dict = model.module.state_dict() if isinstance(model, DataParallel) or isinstance(model,
                                                                                                DDP) else model.state_dict()
        torch.save(state_dict, model_path)


def valid_experts(dataloader, model):
    training = model.training
    model.eval()

    ground_truth = torch.Tensor().cuda()
    predictions = torch.Tensor().cuda()
    exp_preds = torch.Tensor().cuda()

    return_dict = {}

    with torch.no_grad():
        for img, label in dataloader:
            img, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True).long()
            outputs = model(img)
            logits = outputs.logits
            # logts: [B, n_experts, n_classes]
            pred = F.softmax(logits, dim=-1)
            exp_preds = torch.cat((exp_preds, pred))
            pred = torch.mean(pred, dim=1)
            ground_truth = torch.cat((ground_truth, label))
            predictions = torch.cat((predictions, pred))

        for i in range(exp_preds.shape[1]):
            acc, f1, auc, ap, bac, sens, spec, prec, mcc, kappa = compute_avg_metrics(ground_truth, exp_preds[:, i, :], avg='micro')
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
        acc, f1, auc, ap, bac, sens, spec, prec, mcc, kappa = compute_avg_metrics(ground_truth, predictions, avg='micro')
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
    model.train(training)
    return return_dict