import os
import h5py
from pathlib import Path
from tqdm import tqdm
import pandas as pd


csv_path = '/mnt/zhen_chen/patches_CAMELYON16/status.csv'
dara_root = '/mnt/zhen_chen/coordinates_CAMELYON16_pruned'
test_csv = '/mnt/zhen_chen/AIEC/4-MIL/reference.csv'
num_levels = 3
df = pd.DataFrame(columns=['slide_id', 'i', 'j', 'patch_size', 'downsample_factor',
                           'base_level', 'cur_level', 'label'])

slide_info = pd.read_csv(csv_path)
label2num = {'normal': 0, 'tumor': 1}

for i in tqdm(range(slide_info.shape[0]), desc='WSI', position=0):
    status = slide_info.iloc[i]['status']
    slide_id = slide_info.iloc[i]['slide_id']
    if status != 'done' or slide_id.startswith('test_'):
        continue
    slide_name = Path(slide_id).stem
    label = label2num[slide_name.split('_')[0]]
    h5_path = os.path.join(dara_root, f"{slide_name}.h5")
    with h5py.File(h5_path, 'r') as f:
        base_level = f.attrs['base_level']
        patch_size = f.attrs['patch_size']
        downsample_factor = f.attrs['downsample_factor']
        df = df._append({'slide_id': slide_id, 'i': 0, 'j': 0, 'patch_size': patch_size, 'downsample_factor': downsample_factor,
                         'base_level': base_level, 'cur_level': 'overview', 'label': label}, ignore_index=True)
        for level in range(num_levels):
            level_name = f'level_{level}'
            level_coords = f[level_name][:]
            for x in range(level_coords.shape[0]):
                for y in range(level_coords.shape[1]):
                    if all(level_coords[x, y] != -1):
                        df = df._append({'slide_id': slide_id, 'i': level_coords[x, y][0], 'j': level_coords[x, y][1], 'patch_size': patch_size, 'downsample_factor': downsample_factor,
                                         'base_level': base_level, 'cur_level': level_name, 'label': label}, ignore_index=True)
    df.to_csv('./camelyon_training.csv', index=False)

for i in tqdm(range(slide_info.shape[0]), desc='WSI', position=0):
    status = slide_info.iloc[i]['status']
    slide_id = slide_info.iloc[i]['slide_id']
    if status != 'done' or not slide_id.startswith('test_'):
        continue
    slide_name = Path(slide_id).stem
    label = label2num[slide_name.split('_')[0]]
    h5_path = os.path.join(dara_root, f"{slide_name}.h5")
    with h5py.File(h5_path, 'r') as f:
        base_level = f.attrs['base_level']
        patch_size = f.attrs['patch_size']
        downsample_factor = f.attrs['downsample_factor']
        df = df._append({'slide_id': slide_id, 'i': 0, 'j': 0, 'patch_size': patch_size, 'downsample_factor': downsample_factor,
                         'base_level': base_level, 'cur_level': 'overview', 'label': label}, ignore_index=True)
        for level in range(num_levels):
            level_name = f'level_{level}'
            level_coords = f[level_name][:]
            for x in range(level_coords.shape[0]):
                for y in range(level_coords.shape[1]):
                    if all(level_coords[x, y] != -1):
                        df = df._append({'slide_id': slide_id, 'i': level_coords[x, y][0], 'j': level_coords[x, y][1], 'patch_size': patch_size, 'downsample_factor': downsample_factor,
                                         'base_level': base_level, 'cur_level': level_name, 'label': label}, ignore_index=True)
    df.to_csv('./camelyon_testing.csv', index=False)
