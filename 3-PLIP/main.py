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
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from dataset import Transforms, PatchDataset
from models import CreateModel
from loss import ProbabilityLoss, BatchLoss, ChannelLoss


def update_ema_variables(student, teacher, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(teacher.parameters(), student.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, train_loader, student_model, teacher_model, optimizer, writer):
    loss_epoch = 0
    probability_loss_func = ProbabilityLoss()
    batch_sim_loss_func = BatchLoss(args.batch_size, args.world_size)
    channel_sim_loss_func = ChannelLoss(args.batch_size, args.world_size)
    ce_loss_func = nn.CrossEntropyLoss()
    cur_iters = 0
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

        cur_iters += 1
        if args.rank == 0 and cur_iters % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")
            if writer is not None:
                writer.log({'loss': loss.item()})

        loss_epoch += loss.item()
    return loss_epoch


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

    train_dataset = PatchDataset(args.csv_path, transforms=transforms)

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

    n_classes = train_dataset.n_classes
    student = CreateModel(n_classes=n_classes, ema=False).cuda()
    teacher = CreateModel(n_classes=n_classes, ema=True).cuda()

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)

    if args.world_size > 1:
        student = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student)
        student = DDP(student, device_ids=[rank])
    
    loss_min = 1e9
    for epoch in range(args.epochs):
        student.train()
        loss_epoch = train(args, train_loader, student, teacher, optimizer, wandb_logger)

        if args.rank == 0:
            print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}")
            if loss_epoch / len(train_loader) < loss_min:
                loss_min = loss_epoch / len(train_loader)
                torch.save(student.module.encoder.state_dict(), f"best_{args.backbone}.pth")


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--visible_gpus', type=str, default='4,5,6,7')
    parser.add_argument('--csv_path', type=str, default='./patch_info.csv')
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

