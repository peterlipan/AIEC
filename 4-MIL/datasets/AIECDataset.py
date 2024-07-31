import os
import torch
import pickle
import pathlib
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


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
    def __init__(self, data_root, csv_file, use_pkl=False, transforms=None):
        super(AIECPyramidDataset, self).__init__()
        self.label2num = {'MMRd': 0, 'NSMP': 1, 'P53abn': 2, 'POLEmut': 3}
        self.num2label = {0: 'MMRd', 1: 'NSMP', 2: 'P53abn', 3: 'POLEmut'}
        self.num_classes = 4
        self.data_root = data_root
        self.slide_idx = csv_file['slide_id'].values
        self.diagnosis = csv_file['diagnosis'].values
        self.labels = csv_file['diagnosis'].map(self.label2num).values
        self.use_pkl = use_pkl
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        suffix = '.pkl' if self.use_pkl else '.pt'
        subfolder = 'pkl_files' if self.use_pkl else 'pt_files'
        wsi_name = self.slide_idx[idx]
        file_path = os.path.join(self.data_root, self.diagnosis[idx], subfolder, self.slide_idx[idx] + suffix)
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
        return wsi_name, features, label

    @staticmethod
    def collate_fn(batch):
        wsi_names, features, labels = zip(*batch)
        # features: [batch_size, seq_len, n_views, n_features] for multiple views
        features = pad_sequence(features, batch_first=True).float()
        labels = torch.tensor(labels).long()
        return wsi_names, features, labels