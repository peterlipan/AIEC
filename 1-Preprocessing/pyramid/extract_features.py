import os
import torch
import pathlib
import argparse
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset_helpers import Whole_Slide_Bag
from feature_extractors import get_encoder


def extract_features(model, dataloader):
    model.eval()
    wsi_features = torch.Tensor().cuda()
    with torch.no_grad():
        for i, (img, level) in enumerate(dataloader):
            img = img.cuda(non_blocking=True)
            features = model(img)
            wsi_features = torch.cat((wsi_features, features))
    return wsi_features


def main(args):
    csv = pd.read_csv(args.csv_path)
    for i in tqdm(range(csv.shape[0]), desc='WSIs'):
        if csv['status'][i] == 'done':
            subtype = csv['subtype'][i]
            slide_id = csv['slide_id'][i]
            slide_name = pathlib.Path(slide_id).stem
            slide_path = os.path.join(args.wsi_dir, subtype, slide_id)
            h5_path = os.path.join(args.h5_dir, subtype, 'patches', slide_id)
            save_path = os.path.join(args.save_dir, subtype, 'pt_files', f'{slide_name}.pt')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            model, transforms = get_encoder(args.backbone, target_size=args.patch_size)
            slide_dataset = Whole_Slide_Bag(slide_path, h5_path, img_transforms=transforms)
            dataloader = DataLoader(slide_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

            wsi_features = extract_features(model, dataloader)
            torch.save(wsi_features, save_path)
            
if '__name__' == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='/mnt/zhen_chen/pyramid_patches/status.csv')
    parser.add_argument('--wsi_dir', type=str, default='/mnt/zhen_chen/AIEC_tiff')
    parser.add_argument('--h5_dir', type=str, default='/mnt/zhen_chen/pyramid_patches')
    parser.add_argument('--save_dir', type=str, default='/mnt/zhen_chen/pyramid_features')
    parser.add_argument('--backbone', type=str, default='resnet50_trunc')
    parser.add_argument('--patch_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    main(args)
            