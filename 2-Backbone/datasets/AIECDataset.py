import os
import torch
import pathlib
import pandas as pd
from torch.utils.data import Dataset


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


class AIECPyramidDataset(Dataset):
    def __init__(self, data_root, csv_file, use_h5=False, transforms=None):
        super(AIECPyramidDataset, self).__init__()
        self.label2num = {'MMRd': 0, 'NSMP': 1, 'P53abn': 2, 'POLEmut': 3}
        self.num2label = {0: 'MMRd', 1: 'NSMP', 2: 'P53abn', 3: 'POLEmut'}
        self.num_classes = 4
        self.data_root = data_root
        self.slide_idx = csv_file['slide_id'].values
        self.diagnosis = csv_file['diagnosis'].values
        self.labels = csv_file['diagnosis'].map(self.label2num).values
        self.use_h5 = use_h5
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        suffix = '.h5' if self.use_h5 else '.pt'
        subfolder = 'h5_files' if self.use_h5 else 'pt_files'
        file_path = os.path.join(self.data_root, self.diagnosis[idx], subfolder, self.slide_idx[idx] + suffix)
        features = torch.load(file_path)
        if self.transforms is not None:
            # if a list of transforms, implement MoE
            if isinstance(self.transforms, list):
                features = torch.stack([transform(features) for transform in self.transforms], dim=1)
            else:
                features = self.transforms(features)
        label = self.labels[idx]
        return features, label