import os
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from WholeSlideImage import WholeSlideImage
import multiprocessing as mp


def process_slide(args, slide_id, dst_dir):
    slide_path = os.path.join(args.src, slide_id)
    try:
        wsi = WholeSlideImage(slide_path, dst_dir, patch_size=args.patch_size, base_downsample=args.base_downsample,
                              downsample_factor=args.downsample_factor, num_levels=args.num_levels, use_otsu=not args.no_use_otsu,
                              sthresh=args.sthresh, sthresh_up=args.sthresh_up, mthresh=args.mthresh, padding=not args.no_padding,
                              visualize=not args.no_visualize, visualize_width=args.visualize_width, skip=not args.no_skip, save_patch=args.save_patch)
        wsi.multi_level_segment()
        return slide_id, 'done'
    except Exception as e:
        print(f'Error processing {slide_id}:')
        print(e)
        return slide_id, 'error'


def init_df(args):
    image_extensions = ['.tif', '.tiff', '.svs', '.mrxs', '.ndpi']
    filenames = [f.name for f in Path(args.src).rglob('*') if f.suffix in image_extensions]
    status = ['tbp'] * len(filenames)
    process = [1] * len(filenames)
    df = pd.DataFrame({'slide_id': filenames, 'status': status, 'process': process})
    return df


def main(args):
    os.makedirs(args.dst, exist_ok=True)
    df_path = os.path.join(args.dst, 'status.csv')
    df = init_df(args)
    df.to_csv(df_path, index=False)

    with mp.Pool(processes=args.workers) as pool:
        results = [pool.apply_async(process_slide, args=(args, slide_id, args.dst)) for slide_id in df['slide_id']]
        pool.close()
        pool.join()
        for i, res in tqdm(enumerate(results), total=len(results)):
            slide_id, status = res.get()
            df.loc[df['slide_id'] == slide_id, 'status'] = status
        df.to_csv(df_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Whole Slide Image Processing')
    parser.add_argument('--src', type=str, default='/vast/palmer/scratch/liu_xiaofeng/xl693/li/CAMELYON16')
    parser.add_argument('--dst', type=str, default='/vast/palmer/scratch/liu_xiaofeng/xl693/li/patches_CAMELYON16')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--base_downsample', type=int, default=20)
    parser.add_argument('--downsample_factor', type=int, default=4)
    parser.add_argument('--num_levels', type=int, default=3)
    parser.add_argument('--no_use_otsu', action='store_true')
    parser.add_argument('--sthresh', type=int, default=20)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--sthresh_up', type=int, default=255)
    parser.add_argument('--mthresh', type=int, default=7)
    parser.add_argument('--no_padding', action='store_true')
    parser.add_argument('--no_visualize', action='store_true')
    parser.add_argument('--visualize_width', type=int, default=1024)
    parser.add_argument('--no_skip', action='store_true')
    parser.add_argument('--save_patch', action='store_true')
    args = parser.parse_args()
    main(args)
