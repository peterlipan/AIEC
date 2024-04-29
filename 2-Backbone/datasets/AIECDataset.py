import os
import torch
import pathlib
import pandas as pd


class AIECDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, csv_file, use_h5=False):
        super(AIECDataset, self).__init__()
        self.label2num = {'MMRD': 0, 'NSMP': 1, 'P53abn': 2, 'POLEmut': 3}
        self.num2label = {0: 'MMRD', 1: 'NSMP', 2: 'P53abn', 3: 'POLEmut'}
        self.num_classes = len(self.label2num)
        self.data_root = data_root
        self.slide_idx = csv_file['slide_id'].values
        self.diagnosis = csv_file['diagnosis'].values
        self.labels = csv_file['diagnosis'].map(self.label2num).values
        self.use_h5 = use_h5
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        suffix = '.h5' if self.use_h5 else '.pt'
        file_path = os.path.join(self.data_root, self.diagnosis[idx], self.slide_idx[idx] + suffix)
        features = torch.load(file_path)
        label = self.labels[idx]
        return features, label

