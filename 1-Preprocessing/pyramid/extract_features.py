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


def extract_features(model, dataloader):
    model.eval()
    wsi_features = torch.Tensor().cuda()
    wsi_levels = []
    with torch.no_grad():
        for i, (img, level) in enumerate(dataloader):
            img = img.cuda(non_blocking=True)
            features = model(img)
            wsi_features = torch.cat((wsi_features, features))
            wsi_levels.extend(level)
    return wsi_features, np.array(wsi_levels)


def main(rank, csv, args):
    args.rank = rank
    args.device = rank
    # fetch the rank-th subtable of the csv
    sub_csv = csv[rank]

    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    torch.cuda.set_device(rank)

    model, transforms = get_encoder(args.backbone, target_img_size=args.patch_size)
    model = model.cuda()
    # parallel
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])

    for i in range(sub_csv.shape[0]):
        if sub_csv.iloc[i]['status'] == 'done':
            subtype = sub_csv.iloc[i]['subtype']
            slide_id = sub_csv.iloc[i]['slide_id']
            slide_name = pathlib.Path(slide_id).stem
            print(f'Processing {slide_name} of subtype {subtype}')

            slide_path = os.path.join(args.wsi_dir, subtype, slide_id)
            h5_path = os.path.join(args.h5_dir, subtype, 'patches', f'{slide_name}.h5')
            save_path = os.path.join(args.save_dir, subtype, 'pt_files', f'{slide_name}.pt')

            if os.path.exists(save_path) and not args.no_skip:
                print(f'{slide_name} already processed, skipping')
                continue
            else:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            

            slide_dataset = Whole_Slide_Bag(slide_path, h5_path, img_transforms=transforms)
            dataloader = DataLoader(slide_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

            wsi_features, wsi_levels = extract_features(model, dataloader)
            level_shapes = slide_dataset.shapes
            save_features = {}
            for level, shape in level_shapes.items():
                save_features[level] = wsi_features[wsi_levels == level].view(shape[0], shape[1], wsi_features.size(-1)).cpu()
            torch.save(save_features, save_path)
            print(f'{slide_name} processed and saved!!')
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='/mnt/zhen_chen/pyramid_patches/status.csv')
    parser.add_argument('--wsi_dir', type=str, default='/mnt/zhen_chen/AIEC_tiff')
    parser.add_argument('--h5_dir', type=str, default='/mnt/zhen_chen/pyramid_patches')
    parser.add_argument('--save_dir', type=str, default='/mnt/zhen_chen/pyramid_features')
    parser.add_argument('--backbone', type=str, default='resnet50_trunc')
    parser.add_argument('--patch_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--no_skip', action='store_true')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = VISIBLE_GPU
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    csv = pd.read_csv(args.csv_path).sample(frac=1).reset_index(drop=True)
    num_gpu = len(VISIBLE_GPU.split(','))
    args.world_size = num_gpu

    # split the csv into num_gpu subtables
    split_dfs = np.array_split(csv, num_gpu)
    split_dfs_list = [pd.DataFrame(split) for split in split_dfs]
    
    mp.spawn(main, args=(split_dfs_list, args), nprocs=num_gpu, join=True)
            