# This is to generate a 'pseudo' csv file from the featuresx20_256 folder
# TODO: generate the csv file with the patient information. Columns: patient_id, slide_id, diagnosis
import os
import torch
import pathlib
import pandas as pd


if __name__ == '__main__':
    groups = ['MMRd', 'NSMP', 'P53abn', 'POLEmut']
    df = pd.DataFrame(columns=['patient_id', 'slide_id', 'num_patches', 'diagnosis'])
    for item in groups:
        subpath = os.path.join('/mnt/zhen_chen/pyramid_features', item, 'pt_files')
        filenames = [f for f in os.listdir(subpath) if f.endswith('.pt')]
        slide_idx = [pathlib.Path(f).stem for f in filenames]
        num_levels = [len(torch.load(os.path.join(subpath, f))) for f in filenames]
        # take the slide_idx as patient_id
        patient_id = slide_idx
        diagnosis = [item] * len(slide_idx)
        df = df._append(pd.DataFrame({'patient_id': patient_id, 'slide_id': slide_idx, 'num_levels': num_levels,
        'diagnosis': diagnosis}))
    df.to_csv('/mnt/zhen_chen/AIEC/2-Backbone/aiec_info.csv', index=False)