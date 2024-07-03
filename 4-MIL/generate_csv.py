# This is to generate a 'pseudo' csv file from the featuresx20_256 folder
# TODO: generate the csv file with the patient information. Columns: patient_id, slide_id, diagnosis
import os
import torch
import pathlib
import pandas as pd


if __name__ == '__main__':
    groups = ['MMRd', 'NSMP', 'P53abn', 'POLEmut']
    df = pd.DataFrame(columns=['patient_id', 'slide_id', 'diagnosis', 'level_0', 'level_1', 'level_2'])
    for item in groups:
        subpath = os.path.join('/mnt/zhen_chen/pyramid_features_512_PLIP', item, 'pt_files')
        filenames = [f for f in os.listdir(subpath) if f.endswith('.pt')]
        for f in filenames:
            features = torch.load(os.path.join(subpath, f))
            slide_idx = pathlib.Path(f).stem
            patient_id = slide_idx
            row = {'patient_id': patient_id, 'slide_id': slide_idx, 'diagnosis': item, 'level_0': 0, 'level_1': 0, 'level_2': 0} 
            for level in ['level_0', 'level_1', 'level_2']:
                level_feature = features[level]
                level_feature = level_feature.view(-1, level_feature.shape[-1])
                non_zero = torch.any(level_feature != 0, dim=-1)
                num_non_zero = torch.sum(non_zero).item()
                row[level] = num_non_zero
            if row['level_0'] < 50:
                continue
            
            df = df._append(row, ignore_index=True)
    df.to_csv('/mnt/zhen_chen/AIEC/4-MIL/aiec_info.csv', index=False)