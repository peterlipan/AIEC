import os
import torch
import pickle
import pathlib
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class AIECPyramidDataset(Dataset):
    def __init__(self, data_root, csv_file, use_pkl=False, transforms=None, task='subtype'):
        super(AIECPyramidDataset, self).__init__()
        self.subtype_enc = {'MMRd': 0, 'NSMP': 1, 'P53abn': 2, 'POLEmut': 3}
        self.grade_enc = {'I': 0, 'II': 1, 'III': 2}
        self.type_enc = {'Carcinosarcoma': 0, 'Clear cell carcinoma': 1,
                          'Endometrioid adenocarcinoma': 2, 'Mixed adenocarcinoma': 3,
                          'Serous carcinoma': 4, 'Undifferentiated carcinoma': 5}
        
        self.data_root = data_root
        self.task = task
        
        if task == 'subtype':
            self.csv_file = csv_file.dropna(subset=['Tumor.MolecularSubtype'])
            self.labels = self.csv_file['Tumor.MolecularSubtype'].map(self.subtype_enc).values
        elif task == 'grade':
            self.csv_file = csv_file.dropna(subset=['Tumor.Grading'])
            self.labels = self.csv_file['Tumor.Grading'].map(self.grade_enc).values
        elif task == 'type':
            self.csv_file = csv_file.dropna(subset=['Tumor.Type'])
            self.labels = self.csv_file['Tumor.Type'].map(self.class_enc).values
        elif task == 'stage':
            self.csv_file = csv_file.dropna(subset=['Tumor.Staging'])
            self.labels = self.csv_file['Tumor.Staging'].map(self.grade_enc).values
        elif task == 'survival':
            self.csv_file = csv_file.dropna(subset=['Overall.Survival.Interval'])
            self.labels = self.csv_file['Overall.Survival.Interval'].values
            self.c = 1 - self.csv_file['Overall.Survival.Status(1: DECEASED; 0: LIVING)'].values
            self.dead = self.csv_file['Overall.Survival.Status(1: DECEASED; 0: LIVING)'].values
            self.event_time = self.csv_file['Overall.Survival.Months'].values * 30
        else:
            raise ValueError(f"Unknown task: {task}")

        self.patient_id = self.csv_file['Case.ID'].values
        self.num_classes = len(set(self.labels))
        self.filenames = self.csv_file['Filename'].values
        self.slide_idx = self.csv_file['Slide.ID'].values
        self.subtype = self.csv_file['Tumor.MolecularSubtype'].values
        self.use_pkl = use_pkl
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        subfolder = 'pkl_files' if self.use_pkl else 'pt_files'
        pid = self.patient_id[idx]
        wsi_name = self.slide_idx[idx]
        file_path = os.path.join(self.data_root, self.subtype[idx], subfolder, self.filenames[idx])
        if self.use_pkl:
            with open(file_path, 'rb') as f:
                features = pickle.load(f)
            features = torch.from_numpy(features)
        else:
            features = torch.load(file_path)
        if self.transforms is not None:
            # if a list of transforms, implement MoE
            if isinstance(self.transforms, list):
                # features: [seq_len, n_views, n_features]
                features = pad_sequence([transform(features) for transform in self.transforms], batch_first=False)
            else:
                # features: [seq_len, n_features]
                features = self.transforms(features)
        label = self.labels[idx]
        c = self.c[idx] if hasattr(self, 'c') else None
        dead = self.dead[idx] if hasattr(self, 'dead') else None
        event_time = self.event_time[idx] if hasattr(self, 'event_time') else None

        return pid, wsi_name, features, label, c, dead, event_time

    @staticmethod
    def collate_fn(batch):
        pid, wsi_names, features, labels, c, dead, event_time = zip(*batch)
        # features: [batch_size, seq_len, n_views, n_features] for multiple views
        features = pad_sequence(features, batch_first=True).float()
        labels = torch.tensor(labels).long()
        return_dict = {
            'patient_id': pid,
            'filename': wsi_names,
            'features': features,
            'label': labels
        }
        if c[0] is not None:
            return_dict['c'] = torch.tensor(c).float()
            return_dict['dead'] = torch.tensor(dead).float()
            return_dict['event_time'] = torch.tensor(event_time).float()

        return return_dict