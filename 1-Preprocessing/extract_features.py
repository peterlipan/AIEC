import os
import torch
import pathlib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from dataset_helpers import Whole_Slide_Bag
from feature_extractors import get_encoder
from torch.nn.parallel import DistributedDataParallel as DDP

VISIBLE_GPU = '0,1,2,3'


def extract_features(model, level_shapes, feature_dim, dataloader):
    model.eval()
    save_features = {}
    wsi_features = torch.Tensor()
    wsi_levels = []
    patch_coord_i = []
    patch_coord_j = []
    for level, shape in level_shapes.items():
        save_features[level] = torch.zeros((shape[0], shape[1], feature_dim))
    save_features['overview'] = torch.zeros((1, 1, feature_dim))

    with torch.no_grad():
        for i, (img, level, coord_i, coord_j) in enumerate(dataloader):
            img = img.cuda(non_blocking=True)
            features = model(img).cpu()
            wsi_features = torch.cat((wsi_features, features))
            wsi_levels.extend(level)
            patch_coord_i.extend(coord_i)
            patch_coord_j.extend(coord_j)
        
        for f, l, i, j in zip(wsi_features, wsi_levels, patch_coord_i, patch_coord_j):
            save_features[l][i, j] = f

    return save_features

def main(rank, csv, args):
    args.rank = rank
    args.device = rank
    # fetch the rank-th subtable of the csv
    sub_csv = csv[rank]

    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    torch.cuda.set_device(rank)

    finetuned_model_path = f"/mnt/zhen_chen/AIEC/3-PLIP/best_{args.backbone}.pth"
    if not os.path.exists(finetuned_model_path):
        print(f"Finetuned model for {args.backbone} not found, using pretrained model")
        finetuned_model_path = ''

    model, feature_dim, transforms = get_encoder(args.backbone, target_img_size=args.patch_size, finetuned=finetuned_model_path)
    model = model.cuda()
    # parallel
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])

    for i in range(sub_csv.shape[0]):
        if sub_csv.iloc[i]['status'] == 'done':
            slide_id = sub_csv.iloc[i]['slide_id']
            slide_name = pathlib.Path(slide_id).stem
            print(f'Processing {slide_name}...')

            slide_path = os.path.join(args.wsi_dir, slide_id)
            coord_path = os.path.join(args.h5_dir, 'coordinates', f'{slide_name}.h5')
            patch_path = os.path.join(args.h5_dir, 'patches', slide_name)
            save_path = os.path.join(args.save_dir, 'pt_files', f'{slide_name}.pt')

            if os.path.exists(save_path) and not args.no_skip:
                print(f'{slide_name} already processed, skipping')
                continue
            else:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

            slide_dataset = Whole_Slide_Bag(slide_path, coord_path, patch_path, img_transforms=transforms)
            dataloader = DataLoader(slide_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
            level_shapes = slide_dataset.shapes
            wsi_features = extract_features(model, level_shapes, feature_dim, dataloader)
            
            torch.save(wsi_features, save_path)
            print(f'{slide_name} processed and saved!!')
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='/vast/palmer/scratch/liu_xiaofeng/xl693/li/patches_CAMELYON16/status.csv')
    parser.add_argument('--wsi_dir', type=str, default='/vast/palmer/scratch/liu_xiaofeng/xl693/li/CAMELYON16')
    parser.add_argument('--h5_dir', type=str, default='/vast/palmer/scratch/liu_xiaofeng/xl693/li/patches_CAMELYON16')
    parser.add_argument('--save_dir', type=str, default='/vast/palmer/scratch/liu_xiaofeng/xl693/li/features_CAMELYON16')
    parser.add_argument('--backbone', type=str, default='densenet121')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--no_skip', action='store_true')
    parser.add_argument('--visible_gpu', type=str, default='0,1,2,3')
    parser.add_argument('--port', type=str, default='12345')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_gpu
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port

    csv = pd.read_csv(args.csv_path).sample(frac=1).reset_index(drop=True)
    num_gpu = len(VISIBLE_GPU.split(','))
    args.world_size = num_gpu

    # split the csv into num_gpu subtables
    split_dfs = np.array_split(csv, num_gpu)
    split_dfs_list = [pd.DataFrame(split) for split in split_dfs]
    
    mp.spawn(main, args=(split_dfs_list, args), nprocs=num_gpu, join=True)
            