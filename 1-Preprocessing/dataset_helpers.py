import os
import h5py
import openslide
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms


class Whole_Slide_Bag(Dataset):
    def __init__(self, wsi_path, coord_path, patch_path, img_transforms=None, mode='patch'):

        self.img_transforms = img_transforms
        self.coord_path = coord_path
        self.patch_path = patch_path
        self.mode = mode

        if mode == 'coordinate':
            self.wsi = openslide.OpenSlide(wsi_path)
            self.shapes = {}
            with h5py.File(self.coord_path, "r") as f:
                self.num_levels = f.attrs['num_levels']
                self.patch_size = f.attrs['patch_size']
                self.downsample_factor = f.attrs['downsample_factor']
                self.base_level = f.attrs['base_level']
                self.coords = []
                self.levels = []
                self.locs = []
                for l in range(self.num_levels):
                    self.shapes[f'level_{l}'] = f[f'level_{l}'].shape
                    level_coords = np.array(f[f'level_{l}'])
                    for i in range(level_coords.shape[0]):
                        for j in range(level_coords.shape[1]):
                            if all(level_coords[i, j] != -1):
                                self.levels.append(f'level_{l}')
                                self.coords.append(level_coords[i, j])
                                self.locs.append([i, j])
            self.levels.append('overview')
            self.coords.append([0, 0])
            self.locs.append([0, 0])
        
        elif mode == 'patch':
            self.shapes = {}
            with h5py.File(self.coord_path, "r") as f:
                self.num_levels = f.attrs['num_levels']
                self.patch_size = f.attrs['patch_size']
                self.downsample_factor = f.attrs['downsample_factor']
                for i in range(self.num_levels):
                    self.shapes[f'level_{i}'] = f[f'level_{i}'].shape

            self.img_paths = []
            self.img_paths.append(os.path.join(patch_path, 'overview.png'))
            for i in range(self.num_levels):
                level_path = os.path.abspath(os.path.join(patch_path, f'level_{i}'))
                self.img_paths.extend([os.path.join(level_path, f) for f in os.listdir(level_path) if f.endswith('.png')])

    def __len__(self):
        if self.mode == 'coordinate':
            return len(self.coords)
        else:
            return len(self.img_paths)

    def __getitem__(self, idx):
        if self.mode == 'coordinate':
            coord = self.coords[idx]
            level = self.levels[idx]
            i, j = self.locs[idx]
            if level == 'overview':
                dimensions = self.wsi.level_dimensions[-1]
                fetch_level = len(self.wsi.level_dimensions) - 1
                img = self.wsi.read_region((0, 0), fetch_level, dimensions).convert('RGB')
                img = self.img_transforms(img)
                return img, 'overview', 0, 0
            level_num = int(level[-1])
            size = int(self.patch_size * self.downsample_factor ** level_num)

            # crop self.img with coords
            img = self.wsi.read_region(coord, self.base_level, (size, size)).convert('RGB')
            # follow DTFD
            img = img.resize((256, 256))
            img = self.img_transforms(img)
            return img, level, i, j
        
        elif self.mode == 'patch':
            img_path = self.img_paths[idx]
            if Path(img_path).stem == 'overview':
                img = Image.open(img_path).convert('RGB')
                img = self.img_transforms(img)
                return img, 'overview', 0, 0
            level = Path(img_path).parts[-2]
            i = int(Path(img_path).name.split('_')[0])
            j = int(Path(img_path).name.split('_')[1])
            img = Image.open(self.img_paths[idx]).convert('RGB')
            img = self.img_transforms(img)
            return img, level, i, j
