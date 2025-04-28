import os
import torch
import pickle
import pathlib
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class Slide:
    def __init__(self, root: str, row: pd.Series, task='subtyping', transforms=None):
        # root: root path to the WSI samples
        # row: a row in the WSI information dataframe
        subtype = row['Tumor.MolecularSubtype']
        self.filename = row['Filename']
        path = os.path.join(root, subtype, 'pt_files', self.filename)
        
        if task == 'grading':
            self.label_enc = {'I': 0, 'II': 1, 'III': 2}
            self.label = self.label_enc[row['Tumor.Grading']]
        elif task == 'subtyping':
            self.label_enc = {'MMRd': 0, 'NSMP': 1, 'P53abn': 2, 'POLEmut': 3}
            self.label = self.label_enc[row['Tumor.MolecularSubtype']]
        elif task == 'survival':
            self.label = row['Overall.Survival.Interval']
        else:
            raise ValueError('Invalid task name. Choose from "grading", "subtyping", or "survival".')
    
        self.event_time = row['Overall.Survival.Months'] * 30
        self.c = 0 if row['Death(Yes or No)']=='Yes' else 1
        self.dead = 1 if row['Death(Yes or No)']=='Yes' else 0
        features = torch.load(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        if transforms is not None:
            # if a list of transforms, implement MoE
            if isinstance(transforms, list):
                # features: [seq_len, n_views, n_features]
                features = pad_sequence([transform(features) for transform in transforms], batch_first=False)
            else:
                # features: [seq_len, n_features]
                features = transforms(features)
        self.features = features

    
    def _to_dict(self):
        label = torch.tensor(self.label).long()
        event_time = torch.tensor(self.event_time).float()
        c = torch.tensor(self.c).float()
        dead = torch.tensor(self.dead).float()
        return {
            'features': self.features,
            'label': label,
            'event_time': event_time,
            'c': c,
            'dead': dead,
        }
        


class AIECDataset(Dataset):
    def __init__(self, data_root, csv_file, use_h5=False):
        super(AIECDataset, self).__init__()
        self.label2num = {'MMRd': 0, 'NSMP': 1, 'P53abn': 2, 'POLEmut': 3}
        self.num2label = {0: 'MMRd', 1: 'NSMP', 2: 'P53abn', 3: 'POLEmut'}
        self.num_classes = 4
        self.data_root = data_root
        self.slide_idx = csv_file['slide_id'].values
        self.diagnosis = csv_file['diagnosis'].values
        self.labels = csv_file['diagnosis'].map(self.label2num).values
        self.use_h5 = use_h5
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        suffix = '.h5' if self.use_h5 else '.pt'
        subfolder = 'h5_files' if self.use_h5 else 'pt_files'
        file_path = os.path.join(self.data_root, self.diagnosis[idx], subfolder, self.slide_idx[idx] + suffix)
        features = torch.load(file_path)
        label = self.labels[idx]
        return features, label
    
    @staticmethod
    def collate_fn(batch):
        features, labels = zip(*batch)
        # features: [batch_size, seq_len, n_views, n_features] for multiple views
        features = pad_sequence(features, batch_first=True).float()
        labels = torch.tensor(labels).long()
        return features, labels


class AIECPyramidDataset(Dataset):
    def __init__(self, root, csv, task='grading',
                 transforms=None):
        super(AIECPyramidDataset, self).__init__()

        self.root = root
        self.task = task

        if task == 'grading':
            self.csv = csv.dropna(subset=['Tumor.Grading'])
            self.n_classes = 3
        elif task == 'subtyping':
            self.csv = csv.dropna(subset=['Tumor.MolecularSubtype'])
            self.n_classes = 4
        elif task == 'survival':
            self.csv = csv.dropna(subset=['Overall.Survival.Interval'])
            self.n_classes = 4

        self.transforms = transforms
        self.n_wsi = self.csv.shape[0]
        self.wsi_list = [Slide(self.root, row, task, transforms) for _, row in self.csv.iterrows()]

    def __len__(self):
        return self.n_wsi
    
    def __getitem__(self, idx):
        wsi = self.wsi_list[idx]
        return wsi._to_dict()

    @staticmethod
    def collate_fn(batch):
        features = [item['features'] for item in batch]
        labels = [item['label'] for item in batch]
        event_time = [item['event_time'] for item in batch]
        c = [item['c'] for item in batch]
        dead = [item['dead'] for item in batch]

        features = pad_sequence(features, batch_first=True)
        labels = torch.stack(labels)
        event_time = torch.stack(event_time)
        c = torch.stack(c)
        dead = torch.stack(dead)

        return {
            'features': features,
            'label': labels,
            'event_time': event_time,
            'c': c,
            'dead': dead,
        }