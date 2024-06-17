import os
from tqdm import tqdm
import pandas as pd


csv_path = '/mnt/zhen_chen/AIEC/4-MIL/aiec_info.csv'
dara_root = '/mnt/zhen_chen/pyramid_patches_512'
num_levels = 3
df = pd.DataFrame(columns=['slide_id', 'diagnosis', 'label', 'path'])

slide_info = pd.read_csv(csv_path)
label2num = {'MMRd': 0, 'NSMP': 1, 'P53abn': 2, 'POLEmut': 3}

for i in tqdm(range(slide_info.shape[0]), desc='WSI', position=0):
    slide_id = slide_info.iloc[i]['slide_id']
    diagnosis = slide_info.iloc[i]['diagnosis']
    label = label2num[diagnosis]
    overview_path = os.path.join(dara_root, diagnosis, 'patches', slide_id, 'overview.png')
    df = df._append({'slide_id': slide_id, 'diagnosis': diagnosis, 'label': label, 'path': overview_path}, ignore_index=True)
    for level in tqdm(range(num_levels), desc='level', position=1, leave=False):
        cur_level = f'level_{level}'
        cur_path = os.path.join(dara_root, diagnosis, 'patches', slide_id, cur_level)
        for patch in tqdm(os.listdir(cur_path), desc='patch', position=2, leave=False):
            patch_path = os.path.join(cur_path, patch)
            df = df._append({'slide_id': slide_id, 'diagnosis': diagnosis, 'label': label, 'path': patch_path}, ignore_index=True)
df.to_csv('./patch_info.csv', index=False)
