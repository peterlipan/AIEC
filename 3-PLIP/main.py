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
from loss import NT_Xent


def train(args, train_loader, model, criterion, optimizer, writer):
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking=True)
        x_j = x_j.cuda(non_blocking=True)

        # positive pair, with encoding
        h_i, h_j, z_i, z_j = model(x_i, x_j)

        loss = criterion(z_i, z_j)
        loss.backward()
        optimizer.step()

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))

        if args.rank == 0 and step % 50 == 0:
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

    model = CreateModel(args.backbone)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criteria = NT_Xent(args.batch_size, args.temperature, args.world_size)

    if args.world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank])
    
    loss_min = 1e9
    for epoch in range(args.epochs):
        model.train()
        loss_epoch = train(args, train_loader, model, criteria, optimizer, wandb_logger)

        if args.rank == 0:
            print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}")
            if loss_epoch / len(train_loader) < loss_min:
                loss_min = loss_epoch / len(train_loader)
                torch.save(model.module.encoder.state_dict(), f"best_{args.backbone}.pth")


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

