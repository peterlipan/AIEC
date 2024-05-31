import os
import argparse
import pandas as pd
from tqdm import tqdm
from WholeSlideImage import WholeSlideImage


def init_df(args):

    df = pd.DataFrame(columns=['slide_id', 'subtype', 'status', 'process'])
    subtype = [f for f in os.listdir(args.src) if os.path.isdir(os.path.join(args.src, f))]
    for s in subtype:
        subtype_path = os.path.join(args.src, s)
        subtype_files = [f for f in os.listdir(subtype_path)]
        for f in subtype_files:
            df = df.append({'slide_id': f, 'subtype': s, 'status': 'tbp', 'process': 1}, ignore_index=True)

    return df


def main(args):
    df = init_df(args)
    df.to_csv(args.dst, index=False)
    for i, row in tqdm(df.iterrows()):
        slide_id = row['slide_id']
        subtype = row['subtype']
        print(f'Processing {slide_id} of subtype {subtype}')
        slide_path = os.path.join(args.src, subtype, slide_id)
        target_path = os.path.join(args.dst, subtype)
        try:
            wsi = WholeSlideImage(slide_path, target_path, patch_size=args.patch_size, base_downsample=args.base_downsample,
                                  downsample_factor=args.downsample_factor, num_levels=args.num_levels, use_otsu=args.use_otsu,
                                  sthresh=args.sthresh, sthresh_up=args.sthresh_up, mthresh=args.mthresh, padding=args.padding,
                                  visualize=args.visualize, visualize_width=args.visualize_width)
            wsi.multi_level_segment()
            df.loc[i, 'status'] = 'done'
        except Exception as e:
            print(f'Error processing {slide_id} of subtype {subtype}')
            print(e)
            df.loc[i, 'status'] = 'error'
        df.to_csv(args.dst, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Whole Slide Image Processing')
    parser.add_argument('--src', type=str, default=None)
    parser.add_argument('--dst', type=str, default=None)
    parser.add_argument('--patch_size', type=int, default=512)
    parser.add_argument('--base_downsample', type=int, default=1)
    parser.add_argument('--downsample_factor', type=int, default=4)
    parser.add_argument('--num_levels', type=int, default=3)
    parser.add_argument('--use_otsu', default=True, action='store_true')
    parser.add_argument('--sthresh', type=int, default=20)
    parser.add_argument('--sthresh_up', type=int, default=255)
    parser.add_argument('--mthresh', type=int, default=7)
    parser.add_argument('--padding', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--visualize_width', type=int, default=1024)
    args = parser.parse_args()
    main(args)