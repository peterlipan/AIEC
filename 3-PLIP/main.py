import os
import torch
import wandb
import pickle
import argparse
import numpy as np
import pandas as pd
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from dataset import Transforms, PatchDataset, CoordinateDataset
import torch.nn.functional as F
from models import CreateModel
from loss import ProbabilityLoss, BatchLoss, ChannelLoss
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, \
    roc_auc_score, precision_score, matthews_corrcoef, cohen_kappa_score, average_precision_score
from imblearn.metrics import sensitivity_score, specificity_score


def update_ema_variables(student, teacher, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(teacher.parameters(), student.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def compute_avg_metrics(ground_truth, activations, avg='micro'):
    ground_truth = ground_truth.cpu().detach().numpy()
    activations = activations.cpu().detach().numpy()
    predictions = np.argmax(activations, -1)
    mean_acc = accuracy_score(y_true=ground_truth, y_pred=predictions)
    f1 = f1_score(y_true=ground_truth, y_pred=predictions, average=avg)
    multi_class = 'ovr'
    # For binary classification
    if activations.shape[1] == 2:
        activations = activations[:, 1]
        multi_class = 'raise'
    try:
        auc = roc_auc_score(y_true=ground_truth, y_score=activations, multi_class=multi_class, average=avg)
    except ValueError as error:
        print('Error in computing AUC. Error msg:{}'.format(error))
        auc = 0
    try:
        ap = average_precision_score(y_true=ground_truth, y_score=activations, average=avg)
    except Exception as error:
        print('Error in computing AP. Error msg:{}'.format(error))
        ap = 0
    bac = balanced_accuracy_score(y_true=ground_truth, y_pred=predictions)
    sens = sensitivity_score(y_true=ground_truth, y_pred=predictions, average=avg)
    spec = specificity_score(y_true=ground_truth, y_pred=predictions, average=avg)
    prec = precision_score(y_true=ground_truth, y_pred=predictions, average=avg)
    mcc = matthews_corrcoef(y_true=ground_truth, y_pred=predictions)
    kappa = cohen_kappa_score(y1=ground_truth, y2=predictions)

    return mean_acc, f1, auc, ap, bac, sens, spec, prec, mcc, kappa


def validate(dataloader, model):
    training = model.training
    model.eval()

    ground_truth = torch.Tensor().cuda()
    predictions = torch.Tensor().cuda()

    with torch.no_grad():
        for img, label in dataloader:
            img, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True).long()
            _, logits = model(img)
            pred = F.softmax(logits, dim=1)
            ground_truth = torch.cat((ground_truth, label))
            predictions = torch.cat((predictions, pred))

        acc, f1, auc, ap, bac, sens, spec, prec, mcc, kappa = compute_avg_metrics(ground_truth, predictions, avg='macro')
    model.train(training)
    return acc, f1, auc, ap, bac, sens, spec, prec, mcc, kappa


def train(args, dataloaders, student_model, teacher_model, optimizer, writer):
    probability_loss_func = ProbabilityLoss()
    batch_sim_loss_func = BatchLoss(args.batch_size, args.world_size)
    channel_sim_loss_func = ChannelLoss(args.batch_size, args.world_size)
    ce_loss_func = nn.CrossEntropyLoss()
    train_loader, test_loader = dataloaders
    for step, ((x_i, x_j), label) in enumerate(train_loader):

        x_i, x_j, label = x_i.cuda(non_blocking=True), x_j.cuda(non_blocking=True), label.cuda(non_blocking=True).long()

        # positive pair, with encoding
        features, logits = student_model(x_i)
        with torch.no_grad():
            teacher_features, teacher_logits = teacher_model(x_j)

        cls_loss = ce_loss_func(logits, label)
        probability_loss = probability_loss_func(logits, teacher_logits)
        batch_sim_loss = batch_sim_loss_func(features, teacher_features)
        channel_sim_loss = channel_sim_loss_func(features, teacher_features)

        loss = cls_loss + 5 * probability_loss + 10 * batch_sim_loss + 10 * channel_sim_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update the teacher model
        update_ema_variables(student_model, teacher_model, alpha=0.999, global_step=step)

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))

        if args.rank == 0 and (step % 50 == 0 or step == len(train_loader) - 1):
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")
            test_acc, test_f1, test_auc, test_ap, test_bac, test_sens, test_spec, test_prec, test_mcc, test_kappa = validate(
                test_loader, student_model)
            if writer is not None:
                writer.log({'test': {'Accuracy': test_acc,
                                     'F1 score': test_f1,
                                     'AUC': test_auc,
                                     'AP': test_ap,
                                     'Balanced Accuracy': test_bac,
                                     'Sensitivity': test_sens,
                                     'Specificity': test_spec,
                                     'Precision': test_prec,
                                     'MCC': test_mcc,
                                     'Kappa': test_kappa},
                            'train': {'loss': loss.item(),}}, )

    return test_acc


def main(rank, args, wandb_logger):

    if rank != 0:
        wandb_logger = None

    args.rank = rank
    args.device = rank

    if args.world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(rank)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    transforms = Transforms(size=args.size)

    train_dataset = CoordinateDataset(args.train_csv_path, args.wsi_root, transforms=transforms)

    # set sampler for parallel training
    if args.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
        pin_memory=True,
    )

    if rank == 0:
        test_dataset = CoordinateDataset(args.test_csv_path, args.wsi_root, transforms=transforms.test_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.workers,
            pin_memory=True,
        )
    else:
        test_loader = None

    dataloaders = (train_loader, test_loader)

    n_classes = 2
    student = CreateModel(n_classes=n_classes, ema=False).cuda()
    teacher = CreateModel(n_classes=n_classes, ema=True).cuda()

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)

    if args.world_size > 1:
        student = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student)
        student = DDP(student, device_ids=[rank])
    
    acc = -1
    for epoch in range(args.epochs):
        student.train()
        test_acc = train(args, dataloaders, student, teacher, optimizer, wandb_logger)

        if args.rank == 0:
            print(f"Epoch [{epoch}/{args.epochs}]\t Accuracy: {test_acc}")
            if test_acc > acc:
                acc = test_acc
                torch.save(student.module.encoder.state_dict(), f"best_{args.backbone}.pth")


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--visible_gpus', type=str, default='0,1,2,3')
    parser.add_argument('--train_csv_path', type=str, default='./camelyon_training.csv')
    parser.add_argument('--test_csv_path', type=str, default='./camelyon_testing.csv')
    parser.add_argument('--wsi_root', type=str, default='/mnt/zhen_chen/CAMELYON16')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    args.world_size = len(args.visible_gpus.split(","))

    # Master address for distributed data parallel
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    if not args.debug:
        wandb.login(key="cb1e7d54d21d9080b46d2b1ae2a13d895770aa29")
        config = dict()
        for k, v in vars(args).items():
            config[k] = v

        wandb_logger = wandb.init(
            project="PLIP",
            config=config,
        )
    else:
        wandb_logger = None

    mp.spawn(main, args=(args, wandb_logger, ), nprocs=args.world_size, join=True)

