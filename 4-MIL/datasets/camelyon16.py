import os
import torch
import pickle
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class CAMELYON16Dataset(Dataset):
    def __init__(self, data_root, csv_file, training=False, transforms=None, use_pkl=False):
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
        self.use_pkl = use_pkl


    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        wsi_name = self.images[idx]
        if self.use_pkl:
            with open(self.paths[idx], 'rb') as f:
                features = pickle.load(f)
            features = torch.from_numpy(features)
        else:
            features = torch.load(self.paths[idx])
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


class DTFDDataset(Dataset):
    def __init__(self, path, csv_path, training=True):
        super().__init__()
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
        self.num_classes = 2
        label2num = {'normal': 0, 'tumor': 1}
        self.samples = list(self.data.keys())

        self.labels = []
        self.features = []
        csv = pd.read_csv(csv_path)

        for sample in self.samples:
            sample_feature = []
            if training:
                dx = sample.split('_')[0]
            else:
                dx = csv.loc[csv['slide_id'] == sample, 'diagnosis'].values[0].lower()
            self.labels.append(label2num[dx])
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

    @staticmethod
    def collate_fn(batch):
        wsi_names, features, labels = zip(*batch)
        features = pad_sequence(features, batch_first=True).float()
        labels = torch.tensor(labels).long()
        return wsi_names, features, labels


# Load each patch from WSI instead of features
# End-to-end training
class CamelyonEnd2End(Dataset):
    def __init__(self, wsi_path, csv_path, training=True, transforms=False):
        super().__init__()
