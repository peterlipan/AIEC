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
        self.shapes = {}

        with h5py.File(self.file_path, "r") as f:
            self.num_levels = f.attrs['num_levels']
            self.patch_size = f.attrs['patch_size']
            self.downsample_factor = f.attrs['downsample_factor']
            self.coords = []
            self.levels = []
            for i in range(self.num_levels):
                coord = np.array(f[f'level_{i}']).reshape(-1, 2).tolist()
                self.shapes[f'level_{i}'] = f[f'level_{i}'].shape
                self.coords.extend(coord)
                self.levels.extend([f'level_{i}']*len(coord))

        # preload wsi at the base level
        self.img = self.wsi.read_region((0, 0), 0, self.wsi.level_dimensions[0]).convert('RGB')


    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        level = self.levels[idx]
        level_num = int(level[-1])
        size = int(self.patch_size * self.downsample_factor ** level_num)

        # crop self.img with coords
        img = self.img.crop((coord[0], coord[1], coord[0]+size, coord[1]+size))

        img = self.img_transforms(img)
        return img, level
