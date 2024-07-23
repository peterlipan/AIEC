import os
import openslide
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
import torchvision.transforms as T
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2


class Transforms:

    def __init__(self, size):
        s = 1
        color_jitter = T.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )

        normalize = T.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

        self.weak_transform = A.Compose(
            [
                A.Resize(height=size, width=size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        self.strong_transform = A.Compose(
            [
                A.Resize(height=size, width=size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.OneOf([
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ], p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
                A.OneOf([
                    A.GridDistortion(),
                    A.ElasticTransform(),
                    A.OpticalDistortion(),
                ], p=0.7),
                # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                # A.Solarize(p=0.2),
                A.GridDropout(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        self.test_transform = T.Compose(
            [
                T.Resize(size=(size, size)),
                T.ToTensor(),
                normalize
            ]
        )

    def __call__(self, x):
        img = np.array(x)
        strong_augmentation = self.strong_transform(image=img)["image"]
        weak_augmentation = self.weak_transform(image=img)["image"]
        return strong_augmentation, weak_augmentation


class PatchDataset(Dataset):
    def __init__(self, csv_path, transforms):
        self.patch_info = pd.read_csv(csv_path)
        self.paths = self.patch_info['path'].values
        self.labels = self.patch_info['label'].values
        self.transforms = transforms
        self.n_classes = len(np.unique(self.labels))

    def __len__(self):
        return self.patch_info.shape[0]
    
    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transforms:
            image = self.transforms(image)
        return image, label


class CoordinateDataset(Dataset):
    def __init__(self, csv_path, wsi_root, transforms, training=True):
        self.csv = pd.read_csv(csv_path)
        self.wsi_root = wsi_root
        self.transform = transforms
        if not training:
            self.csv = self.csv.sample(n=2024)

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, idx):
        row = self.csv.iloc[idx]
        slide_id = row['slide_id']
        i = row['i']
        j = row['j']
        base_level = row['base_level']
        level = row['cur_level']
        label = row['label']
        downsample_factor = row['downsample_factor']
        patch_size = row['patch_size']
        slide_path = os.path.join(self.wsi_root, slide_id)
        slide = openslide.OpenSlide(slide_path)
        if level == 'overview':
            dimensions = slide.level_dimensions[-1]
            fetch_level = len(slide.level_dimensions) - 1
        else:
            dimensions = patch_size * downsample_factor ** int(level.split('_')[-1])
            dimensions = (dimensions, dimensions)
            fetch_level = base_level
        img = slide.read_region((i, j), fetch_level, dimensions).convert('RGB')
        img = self.transform(img)
        return img, label
