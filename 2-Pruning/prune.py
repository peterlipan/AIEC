import os
import cv2
import torch
import pathlib
import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from segmentation_model import UNetModel
from patch_tree import PatchTree, LevelPatchDataset
from torch.nn.parallel import DistributedDataParallel as DDP

VISIBLE_GPU = '0,1,2,3'


def inference(model, images):
    model.eval()
    with torch.no_grad():
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        probs = F.interpolate(
                probs,
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            )
        probs = probs.permute(0, 2, 3, 1) # to NHWC
    return probs

def save_heatmaps(images, image_names, probs, save_dir):
    for image, image_name, prob in zip(images, image_names, probs):
        image = image.cpu().numpy().transpose(1, 2, 0)
        image = (image * 255).astype(np.uint8)

        # prob of foreground
        prob = prob[..., 1].cpu().numpy()
        prob = (prob * 255).astype(np.uint8)

        output_path = os.path.join(save_dir, image_name)
        heatmap = cv2.applyColorMap(prob, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
        cv2.imwrite(output_path, overlay)


def main(rank, csv, args):
    args.rank = rank
    args.device = rank
    sub_csv = csv[rank]

    dist.init_process_group("gloo", rank=rank, world_size=args.world_size)
    torch.cuda.set_device(rank)

    # Load model
    model = UNetModel(num_input_channels=3, num_output_channels=2, encoder='resnet50', decoder_block=[3])
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(rank)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])

    for i in range(sub_csv.shape[0]):
        if sub_csv.iloc[i]['status'] == 'done':

            slide_id = sub_csv.iloc[i]['slide_id']
            slide_name = pathlib.Path(slide_id).stem
            print(f'Processing {slide_name}...')

            patch_path = os.path.join(args.root, 'patches', slide_name)
            coord_path = os.path.join(args.root, 'coordinates', f'{slide_name}.h5')
            if args.save:
                heatmap_dir = os.path.join(args.root, 'heatmap', slide_name)
                os.makedirs(heatmap_dir, exist_ok=True)

            tree = PatchTree(coord_path, patch_path, mode=args.mode)
            num_patches = 0
            num_pruned = 0
            for level_id in reversed(range(tree.num_levels)):

                threshold = args.threshold / tree.downsample_factor ** level_id ** 2

                if args.save:
                    save_dir = os.path.join(heatmap_dir, f'level_{level_id}')
                    os.makedirs(save_dir, exist_ok=True)

                level_dataset = LevelPatchDataset(tree, level_id, patch_size=args.patch_size, mode=args.mode)
                num_patches += len(level_dataset)
                level_dataloader = DataLoader(level_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

                for images, filenames, node_idx in level_dataloader:
                    images = images.to(rank)
                    probs = inference(model, images)

                    tissue_probs = probs[..., 1]
                    tissue_area = torch.mean(tissue_probs, dim=(1, 2))

                    prune_mask = tissue_area < threshold
                    prune_nodes = [level_dataset.node_list[idx] for idx in np.array(node_idx)[prune_mask.cpu().tolist()]]
                    num_pruned += len(prune_nodes)
                    for node in prune_nodes:
                        node.delete()

                    if args.save:
                        save_mask = tissue_area >= threshold
                        save_images = images[save_mask]
                        save_names = np.array(filenames)[save_mask.cpu().tolist()]
                        save_probs = probs[save_mask]
                        save_heatmaps(save_images, save_names, save_probs, save_dir)

            if args.mode == 'coordinate':
                tree.save_changes()

            print(f'Pruned {num_pruned} out of {num_patches} patches for WSI {slide_name}!')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--csv_path', type=str, default='/mnt/zhen_chen/patches_AIEC/MMRd/status.csv')
    # url: https://tiatoolbox.dcs.warwick.ac.uk/models/seg/fcn-tissue_mask.pth
    args.add_argument('--model_path', type=str, default='./fcn-tissue_mask.pth')
    args.add_argument('--root', type=str, default='/mnt/zhen_chen/patches_AIEC/MMRd')
    args.add_argument('--save', action='store_true')
    args.add_argument('--batch_size', type=int, default=128)
    args.add_argument('--patch_size', type=int, default=256)
    args.add_argument('--workers', type=int, default=0)
    args.add_argument('--threshold', type=float, default=0.1, help='Threshold of tissue area at the lowest level')
    args.add_argument('--mode', type=str, default='patch')
    args = args.parse_args()

    csv = pd.read_csv(args.csv_path).sample(frac=1).reset_index(drop=True)
    num_gpu = len(VISIBLE_GPU.split(','))
    args.world_size = num_gpu

    os.environ['CUDA_VISIBLE_DEVICES'] = VISIBLE_GPU
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # split the csv into num_gpu subtables
    split_dfs = np.array_split(csv, num_gpu)
    split_dfs_list = [pd.DataFrame(split) for split in split_dfs]

    mp.spawn(main, args=(split_dfs_list, args), nprocs=num_gpu, join=True)
