import os
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
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Solarize(p=0.2),
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
    

    def __len__(self):
        return self.patch_info.shape[0]
    
    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transforms:
            image = self.transforms(image)
        return image, label
