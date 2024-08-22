import os
import pandas as pd


src = '/mnt/zhen_chen/AIEC_rawdata'
csv = pd.read_csv('./tongji_slide_idx.csv')
sub_folders = ['1444', '21444']
modals = ['PDL-1', 'CD3', 'CD8']
patient_modal = []
patient_id_list = []
missing_df = pd.DataFrame(columns=['patient_id', 'diagnosis'])
exist_df = pd.DataFrame(columns=['patient_id', 'diagnosis', 'cohort', 'stain'])


for i in range(csv.shape[0]):
    find = False
    patient_id = csv.iloc[i]['patient_id']
    dx = csv.iloc[i]['diagnosis']
    patient_stain = None
    cor = None
    for sub in sub_folders:
        if find:
            break
        for modal in modals:
            src_path = os.path.join(src, sub, dx, modal)
            ids = os.listdir(src_path)
            if patient_id in ids:
                find = True
                patient_stain = modal
                cor = sub
                break
            elif patient_id.upper() in ids:
                find = True
                patient_id = patient_id.upper()
                patient_stain = modal
                cor = sub
                break
    if find:
        exist_df = exist_df._append({'patient_id': patient_id, 'diagnosis': dx, 'cohort': cor, 'stain': patient_stain}, ignore_index=True)
    else:
        print('No such patient: ', patient_id, ' ', dx)
        missing_df = missing_df._append({'patient_id': patient_id, 'diagnosis': dx}, ignore_index=True)

missing_df = missing_df.drop_duplicates()
exist_df = exist_df.drop_duplicates()
missing_df.to_csv('./missing_samples.csv', index=False)
exist_df.to_csv('./tongji_samples.csv', index=False)

