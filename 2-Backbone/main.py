import os
import torch
import wandb
import pickle
import argparse
import numpy as np
import pandas as pd
from models import MambaMIL
from datasets import AIECDataset
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup
from utils import yaml_config_hook, train, get_optim
import warnings
from sklearn.model_selection import KFold


def main(gpu, args, wandb_logger):
    if gpu != 0:
        wandb_logger = None

    rank = args.nr * args.gpus + gpu
    args.rank = rank
    args.device = rank

    if args.world_size > 1 and not args.dataparallel:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(rank)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load data file
    csv_file = pd.read_csv(args.csv_path)
    kf = KFold(n_splits=args.KFold, shuffle=True, random_state=args.seed)
    # split the dataset based on patients
    for i, (train_id, test_id) in enumerate(kf.split(csv_file['patient_id'].values)):
        # run only on one fold
        if args.fold is not None and i != args.fold:
            continue
        train_patient_idx = csv_file['patient_id'].values[train_id]
        test_patient_idx = csv_file['patient_id'].values[test_id]
        train_csv = csv_file[csv_file['patient_id'].isin(train_patient_idx)]
        test_csv = csv_file[csv_file['patient_id'].isin(test_patient_idx)]

        train_dataset = AIECDataset(args.data_root, train_csv, use_h5=False)
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
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )
        if rank == 0:
            test_dataset = AIECDataset(args.data_root, test_csv, use_h5=False)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        else:
            test_loader = None

        loaders = (train_loader, test_loader)

        num_classes = train_dataset.num_classes

        model = MambaMIL(in_dim=args.feature_dim, n_classes=num_classes, dropout=args.dropout, d_model=args.d_model,
        act=args.activation, aggregation=args.agg, layer=args.num_layers, backbone=args.backbone)

        model.relocate()

        optimizer = get_optim(model, args)
        criteria = nn.CrossEntropyLoss().cuda()
        # scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_epochs * step_per_epoch, args.epochs * step_per_epoch)
        scheduler = None
        
        if args.dataparallel:
            model = convert_model(model)
            model = DataParallel(model, device_ids=[int(x) for x in args.visible_gpus.split(",")])

        else:
            if args.world_size > 1:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model = DDP(model, device_ids=[gpu])
        
        train(loaders, model, criteria, optimizer, scheduler, args, wandb_logger)


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    yaml_config = yaml_config_hook("./config.yaml")
    for k, v in yaml_config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    parser.add_argument('--debug', action="store_true", help='debug mode(disable wandb)')
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes

    # Master address for distributed data parallel
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # check checkpoints path
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    # init wandb
    if not args.debug:
        wandb.login(key="cb1e7d54d21d9080b46d2b1ae2a13d895770aa29")
        config = dict()

        for k, v in yaml_config.items():
            config[k] = v

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
