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
    def __init__(self, wsi_path, coord_path, patch_path, img_transforms=None, file_format='patches'):

        
        self.img_transforms = img_transforms
        self.coord_path = coord_path
        self.patch_path = patch_path
        self.format = file_format

        if file_format == 'coordinates':
            self.wsi = openslide.OpenSlide(wsi_path)
            self.shapes = {}
            with h5py.File(self.coord_path, "r") as f:
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
        
        elif file_format == 'patches':
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
        return len(self.img_paths)

    def __getitem__(self, idx):
        if self.format == 'coordinates':
            coord = self.coords[idx]
            level = self.levels[idx]
            level_num = int(level[-1])
            size = int(self.patch_size * self.downsample_factor ** level_num)

            # crop self.img with coords
            img = self.img.crop((coord[0], coord[1], coord[0]+size, coord[1]+size))
            img = self.img_transforms(img)
            return img, level
        
        elif self.format == 'patches':
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
            

        
