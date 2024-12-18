import os
import torch
import wandb
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from models import create_model
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup
from utils import yaml_config_hook, train, get_optim, convert_model, train_experts
from sklearn.model_selection import KFold
from datasets import AIECPyramidDataset, get_train_transforms, get_test_transforms, experts_train_transforms, experts_test_transforms, CAMELYON16Dataset


def main(gpu, args, wandb_logger):
    if gpu != 0:
        wandb_logger = None

    rank = args.nr * args.gpus + gpu
    args.rank = rank
    args.device = rank

    if args.world_size > 1 and not args.dataparallel:
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    if 'expert' in args.backbone.lower() or 'agents' in args.backbone.lower() or 'tree' in args.backbone.lower():
        train_transforms = experts_train_transforms(n_experts=args.n_experts, num_levels=args.num_levels, downsample_factor=args.downsample_factor, 
        lowest_level=args.lowest_level, dropout=args.tree_dropout, visible_levels=args.visible_levels, fix_agent=args.fix_agent, random_layer=args.random_layer)
        test_transforms = experts_test_transforms(n_experts=args.n_experts, num_levels=args.num_levels, downsample_factor=args.downsample_factor, lowest_level=args.lowest_level, visible_levels=args.visible_levels)
    else:
        train_transforms = get_train_transforms(num_levels=args.num_levels, downsample_factor=args.downsample_factor, lowest_level=args.lowest_level, dropout=args.tree_dropout, visible_levels=args.visible_levels)
        test_transforms = get_test_transforms(num_levels=args.num_levels, downsample_factor=args.downsample_factor, lowest_level=args.lowest_level, visible_levels=args.visible_levels)

    kf = KFold(n_splits=args.KFold, shuffle=True, random_state=args.seed)
    csv = pd.read_csv(args.csv_path) if args.csv_path else None
    for fold, (train_idx, test_idx) in enumerate(kf.split(csv)):
        if fold != args.fold:
            continue

        train_csv = pd.read_csv(args.train_csv) if args.train_csv else csv.iloc[train_idx]
        test_csv = pd.read_csv(args.test_csv) if args.test_csv else csv.iloc[test_idx]

        train_dataset = AIECPyramidDataset(args.data_root, train_csv, use_pkl=False, transforms=train_transforms)
        step_per_epoch = len(train_dataset) // (args.batch_size * args.world_size)

        # set sampler for parallel training
        if args.world_size > 1 and not args.dataparallel:
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
            collate_fn=train_dataset.collate_fn,
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )
        if rank == 0:
            test_dataset = AIECPyramidDataset(args.data_root, test_csv, use_pkl=False, transforms=test_transforms)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn,
            num_workers=args.workers, pin_memory=True)
        else:
            test_loader = None

        loaders = (train_loader, test_loader)
        args.num_classes = train_dataset.num_classes

        model = create_model(args)
        model = model.cuda()

        optimizer = get_optim(model, args)
        criteria = nn.CrossEntropyLoss().cuda()
        if args.scheduler:
            scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_epochs * step_per_epoch, args.epochs * step_per_epoch)
        else:
            scheduler = None
            
        if args.dataparallel:
            model = convert_model(model)
            model = DataParallel(model, device_ids=[int(x) for x in args.visible_gpus.split(",")])

        else:
            if args.world_size > 1:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model = DDP(model, device_ids=[gpu])
        if 'expert' in args.backbone.lower():
            train_experts(loaders, model, criteria, optimizer, scheduler, args, wandb_logger)
        else:
            train(loaders, model, criteria, optimizer, scheduler, args, wandb_logger)


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    yaml_config = yaml_config_hook("./configs/aiec.yaml")
    for k, v in yaml_config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    parser.add_argument('--debug', action="store_true", help='debug mode(disable wandb)')
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes
    args.tree_dropout = [float(x) for x in args.tree_dropout.split(", ")]
    args.downsample_factor = [int(x) for x in args.downsample_factor.split(", ")]

    # Master address for distributed data parallel
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12344'

    # check checkpoints path
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    # init wandb
    if not args.debug:
        wandb.login(key="cb1e7d54d21d9080b46d2b1ae2a13d895770aa29")
        config = vars(args)

        wandb_logger = wandb.init(
            project="AIEC",
            config=config
        )
    else:
        wandb_logger = None

    if args.world_size > 1 and not args.dataparallel:
        print(
            f"Training with {args.world_size} GPUS, waiting until all processes join before starting training"
        )
        mp.spawn(main, args=(args, wandb_logger,), nprocs=args.world_size, join=True)
    else:
        main(0, args, wandb_logger)
