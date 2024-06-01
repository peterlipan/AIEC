import h5py
import openslide
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Whole_Slide_Bag(Dataset):
    def __init__(self, wsi_path, file_path, img_transforms=None):
        self.wsi = openslide.OpenSlide(wsi_path)
        self.img_transforms = img_transforms

        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            self.num_levels = f.attrs['num_levels']
            self.patch_size = f.attrs['patch_size']
            self.downsample_factor = f.attrs['downsample_factor']
            self.coords = []
            self.levels = []
            for i in range(self.num_levels):
                coord = f[f'level_{i}'][:]
                self.coords.append(coord)
                self.levels.append([f'level_{i}']*len(coord))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        level = self.levels[idx]
        size = self.patch_size * self.downsample_factor ** level

        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

        img = self.img_transforms(img)
        return img, level
