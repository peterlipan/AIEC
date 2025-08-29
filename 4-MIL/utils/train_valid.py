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
from .metrics import compute_avg_metrics, compute_surv_metrics
from .losses import CrossSampleConsistency, CrossViewConsistency


def train(dataloaders, model, criteria, optimizer, scheduler, args, logger):
    train_loader, test_loader = dataloaders
    model.train()
    start = time.time()
    xview_criteria = CrossViewConsistency(batch_size=args.batch_size, world_size=args.world_size)
    cur_iter = 0

    accumulation_steps = getattr(args, 'accumulation_steps', 1)

    for epoch in range(args.epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        for i, data in enumerate(train_loader):
            data = {k: v.cuda(non_blocking=True) if hasattr(v, 'cuda') else v for k, v in data.items()}
            outputs = model(data)
            logits, agent_features = outputs.logits, outputs.agent_features            

            # classification loss
            cls_loss = criteria(outputs, data)

            if agent_features is not None:
                xview_loss = args.lambda_xview * xview_criteria(agent_features, data['label'])
                loss = cls_loss + xview_loss
            else:
                loss = cls_loss

            # Normalize loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

            if dist.is_available() and dist.is_initialized():
                loss = loss.data.clone()
                dist.all_reduce(loss.div_(dist.get_world_size()))

            cur_iter += 1
            if args.rank == 0:
                train_loss = loss.item() * accumulation_steps  # unnormalize for logging
                cls_loss_value = cls_loss.item()
                xview_loss_value = xview_loss.item() if xview_loss is not None else 0

                if cur_iter % 200 == 0:
                    cur_lr = optimizer.param_groups[0]['lr']
                    test_dict = validate(test_loader, model, criteria, args.task)
                    if logger is not None:
                        logger.log({'test': test_dict,
                                    'train': {'loss': train_loss,
                                              'cls_loss': cls_loss_value,
                                              'xview_loss': xview_loss_value,
                                              'learning_rate': cur_lr}}, )

                    print('Epoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.6f || Loss: %.4f' % (
                        epoch, args.epochs, i + 1, len(train_loader), time.time() - start,
                        cur_lr, train_loss))

    # # Final validation and model saving
    # if args.rank == 0:
    #     test_dict = validate(test_loader, model, criteria)
    #     if logger is not None:
    #         logger.log({'test': test_dict})

    #     test_acc = test_dict['Accuracy']
    #     model_path = os.path.join(args.checkpoints, f"fold_{args.fold}_acc_{test_acc}.pth")
    #     state_dict = model.module.state_dict() if isinstance(model, (DataParallel, DDP)) else model.state_dict()
    #     torch.save(state_dict, model_path)


def validate(dataloader, model, criterion, task):
    training = model.training
    model.eval()
    loss = 0.0     

        
    if task == 'survival':
        event_indicator = torch.Tensor().cuda() # whether the event (death) has occurred
        event_time = torch.Tensor().cuda()
        estimate = torch.Tensor().cuda() 
    else:
        ground_truth = torch.Tensor().cuda()
        probabilities = torch.Tensor().cuda()
          

    with torch.no_grad():
        for data in dataloader:
            data = {k: v.cuda(non_blocking=True) if hasattr(v, 'cuda') else v for k, v in data.items()}
            outputs = model(data)

            loss += criterion(outputs, data).item()
                
            if task == 'survival':
                risk = -torch.sum(outputs['surv'], dim=1)
                event_indicator = torch.cat((event_indicator, data['dead']), dim=0)
                event_time = torch.cat((event_time, data['event_time']), dim=0)
                estimate = torch.cat((estimate, risk), dim=0)
            else:
                
                prob = outputs.y_prob
                ground_truth = torch.cat((ground_truth, data['label']), dim=0)
                probabilities = torch.cat((probabilities, prob), dim=0)
                        
        if task == 'survival':
            metric_dict = compute_surv_metrics(event_indicator, event_time, estimate)
        else:
            metric_dict = compute_avg_metrics(ground_truth, probabilities)
        metric_dict['Loss'] = loss / len(dataloader)
    
    model.train(training)

    return metric_dict


def fold_univariate_cox_regression_analysis(fold, args, model, dataloader):

    training = model.training
    model.eval()

    event_indicator = torch.empty(0).cuda()
    event_time = torch.empty(0).cuda()
    risk_factor = torch.empty(0).cuda()
    filename = []
    patient_id = []

    df_name = f"{args.KFold}Fold_Cox.xlsx"
    res_path = args.results
    df_path = os.path.join(res_path, df_name)

    if not os.path.exists(res_path):
        os.makedirs(res_path)

    with torch.no_grad():
        for data in dataloader:
            data = {k: v.cuda(non_blocking=True) if hasattr(v, 'cuda') else v for k, v in data.items()}
            outputs = model(data)
            risk = torch.sum(outputs['hazards'], dim=1)
            event_indicator = torch.cat((event_indicator, data['dead']), dim=0)
            event_time = torch.cat((event_time, data['event_time']), dim=0)
            risk_factor = torch.cat((risk_factor, risk), dim=0)
            filename.extend(data['filename'])
            patient_id.extend(data['patient_id'])

    event_indicator = event_indicator.cpu().numpy()
    event_time = event_time.cpu().numpy()
    risk_factor = risk_factor.cpu().numpy()

    fold_df = pd.DataFrame({
        'Case.ID': patient_id,
        'Filename': filename,
        'Fold': [fold] * len(filename),
        'event': event_indicator,
        'duration': event_time,
        f'{args.backbone}': risk_factor,
    })

    # If file exists, read and merge (with overwrite for duplicates)
    if os.path.exists(df_path):
        existing_df = pd.read_excel(df_path)

        # Concatenate and drop duplicates — keeping latest (new fold) entries
        combined_df = pd.concat([existing_df, fold_df], ignore_index=True)
        combined_df.drop_duplicates(subset='Filename', keep='last', inplace=True)
    else:
        combined_df = fold_df

    # Save the combined dataframe
    combined_df.to_excel(df_path, index=False)

    model.train(training)


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
    path = './results.csv'
    if not os.path.exists(path):
        df = pd.DataFrame({'WSI': names, 'Label': labels, f'Epoch_{epoch}': preds})
    else:
        df = pd.read_csv(path)
        assert names == df['WSI'].tolist(), 'WSI names do not match'
        assert labels == df['Label'].tolist(), 'Labels do not match'
        df[f'Epoch_{epoch}'] = preds
    df.to_csv(path, index=False)
