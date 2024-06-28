import os
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset


class CAMELYON16Dataset(Dataset):
    def __init__(self, data_root, csv_file, training=False, transforms=None):
        super().__init__()
        self.training = training
        label2num = {'normal': 0, 'tumor': 1}
        self.num_classes = 2
        if training:
            self.images = [f for f in os.listdir(data_root) if not f.startswith('test')]
            self.labels = [label2num[item.split('_')[0]] for item in self.images]
        else:
            csv = pd.read_csv(csv_file)
            self.images = [f for f in os.listdir(data_root) if f.startswith('test')]
            slide_idx = [Path(f).stem for f in self.images]
            diagnosis = [csv.loc[csv['slide_id'] == slide, 'diagnosis'].values[0].lower() for slide in slide_idx]
            self.labels = [label2num[item] for item in diagnosis]

        self.data_root = data_root
        self.paths = [os.path.join(data_root, item) for item in self.images]
        self.transforms = transforms


    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        wsi_name = self.images[idx]
        features = torch.load(self.paths[idx])
        if self.transforms is not None:
            # if a list of transforms, implement MoE
            if isinstance(self.transforms, list):
                features = torch.stack([transform(features) for transform in self.transforms], dim=0)
            else:
                features = self.transforms(features)
        label = self.labels[idx]
        return wsi_name, features, label
