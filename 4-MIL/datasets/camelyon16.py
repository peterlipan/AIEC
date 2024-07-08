import os
import torch
import pickle
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


class DTFDDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
        self.num_classes = 2
        self.samples = list(self.data.keys())

        self.labels = []
        self.features = []

        for sample in self.samples:
            sample_feature = []
            self.labels.append(self.data[sample][0]['label'])
            for item in self.data[sample]:
                sample_feature.append(torch.from_numpy(item['feature']))
            self.features.append(torch.stack(sample_feature, dim=0))

                

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        name = self.samples[idx]
        feature = self.features[idx]
        label = self.labels[idx]
        return name, feature, label
