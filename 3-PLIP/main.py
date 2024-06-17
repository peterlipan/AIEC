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

    if 'expert' in args.backbone.lower():
        train_transforms = experts_train_transforms(n_experts=args.n_experts, num_levels=args.num_levels, downsample_factor=args.downsample_factor, lowest_level=args.lowest_level)
        test_transforms = experts_test_transforms(n_experts=args.n_experts, num_levels=args.num_levels, downsample_factor=args.downsample_factor, lowest_level=args.lowest_level)
    else:
        train_transforms = get_train_transforms(num_levels=args.num_levels, downsample_factor=args.downsample_factor, lowest_level=args.lowest_level)
        test_transforms = get_test_transforms(num_levels=args.num_levels, downsample_factor=args.downsample_factor, lowest_level=args.lowest_level)

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

        train_dataset = AIECPyramidDataset(args.data_root, train_csv, use_h5=False, transforms=train_transforms)
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
            test_dataset = AIECPyramidDataset(args.data_root, test_csv, use_h5=False, transforms=test_transforms)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
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
    parser.add_argument('--visible_gpus', type=str, default='0,1,2,3,4,5,6,7')

    args = parser.parse_args()

    args.world_size = len(args.visible_gpus.split(","))

    # Master address for distributed data parallel
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # check checkpoints path
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    mp.spawn(main, args=(args,), nprocs=args.world_size, join=True)

