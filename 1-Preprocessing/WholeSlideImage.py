import os
import cv2
import h5py
import time
import pathlib
import openslide
import numpy as np
from PIL import Image
from typing import Union
from skimage.filters import threshold_otsu


class WholeSlideImage(object):
    def __init__(self, src: str, dst: str, patch_size: str=512, base_downsample: int=1, downsample_factor: Union[str,int,list]=[2, 5], num_levels: int=3,
                 use_otsu: bool=True, sthresh: int=20, sthresh_up: int=255, mthresh: int=7, padding: bool=True, visualize: bool=True,
                 visualize_width: int=1024, skip: bool=True, save_patch: bool=False, style: str='DTFD'):
        self.src = src
        self.dst = dst
        # patch size: the width/height of the patch at the base_level
        # not at the level with highest resolutions (level 0)
        # p_size (@base_level) = p_size (@level 0) * base_downsample
        self.patch_size = patch_size
        # To a list
        if isinstance(downsample_factor, int):
            self.downsample_factor = [downsample_factor] * (num_levels - 1) 
        elif isinstance(downsample_factor, str):
            self.downsample_factor = [int(i) for i in downsample_factor.split(',')]
        elif isinstance(downsample_factor, list):
            self.downsample_factor = downsample_factor
        else:
            raise ValueError(f'Unknown type of downsample_factor: {downsample_factor}')
        assert len(self.downsample_factor) == num_levels - 1, "The length of downsample_factor should be num_levels - 1"
        self.num_levels = num_levels
        self.use_otsu = use_otsu
        self.sthresh = sthresh
        self.sthresh_up = sthresh_up
        self.mthresh = mthresh

        self.wsi_name = pathlib.Path(src).stem
        self.wsi = openslide.OpenSlide(src)
        self.level_count = self.wsi.level_count
        self.level_dimensions = self.wsi.level_dimensions
        self.level_downsamples = self._assertLevelDownsamples()
        self.base_downsample = base_downsample
        if self.level_count == 1:
            self.base_level = 0
        else:
            self.base_level = self.wsi.get_best_level_for_downsample(base_downsample)
        self.base_dimensions = self.level_dimensions[self.base_level]
        self.padding = padding
        self.visualize = visualize
        self.visualize_width = visualize_width
        self.skip = skip
        self.save_patch = save_patch
        self.style = style
        self.palette = [(173, 216, 230, 255), (255, 182, 193, 255), (152, 251, 152, 255), (230, 230, 250, 255),
                        (255, 255, 0, 255), (255, 165, 0, 255), (255, 0, 255, 255), (64, 224, 208, 255),
                        (168, 168, 120, 255), (210, 105, 30, 255), (255, 199, 0, 255), (138, 54, 15, 255)]

    def _assertLevelDownsamples(self):
        # estimate the downsample factor for each level, following CLAM
        level_downsamples = []
        dim_0 = self.level_dimensions[0]

        for downsample, dim in zip(self.wsi.level_downsamples, self.wsi.level_dimensions):
            estimated_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))

            level_downsamples.append(estimated_downsample) if estimated_downsample != (
                downsample, downsample) else level_downsamples.append((downsample, downsample))

        return level_downsamples

    def _visualize_grid(self, img, asset_dict, stop_x, stop_y):
        scale = self.level_downsamples[self.base_level][0]
        save_path = os.path.join(self.dst, 'visualization', f'{self.wsi_name}.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        height, width, _ = img.shape
        new_height = int(self.visualize_width * height / width)
        resized_img = cv2.resize(img, (self.visualize_width, new_height), interpolation=cv2.INTER_CUBIC)
        resized_height, resized_width, _ = resized_img.shape
        scaled_stop_x = int(stop_x / scale / width * resized_width)
        scaled_stop_y = int(stop_y / scale / height * resized_height)

        for level in range(self.num_levels):
            grid_x, grid_y = asset_dict[f'level_{level}'][:, :, 0], asset_dict[f'level_{level}'][:, :, 1]
            scaled_grid_x = grid_x[:, 0] / scale / width * resized_width
            scaled_grid_y = grid_y[0] / scale / height * resized_height

            scaled_start_x = int(min(scaled_grid_x))
            scaled_start_y = int(min(scaled_grid_y))

            for x in set(scaled_grid_x):
                cv2.line(resized_img, (int(x), scaled_start_y), (int(x), scaled_stop_y-1), self.palette[level], 2 ** level)

            for y in set(scaled_grid_y):
                cv2.line(resized_img, (scaled_start_x, int(y)), (scaled_stop_x-1, int(y)), self.palette[level], 2 ** level)

            # draw the end line
            cv2.line(resized_img, (scaled_stop_x-1, scaled_start_y), (scaled_stop_x-1, scaled_stop_y-1), self.palette[level], 2 ** level)
            cv2.line(resized_img, (scaled_start_x, scaled_stop_y-1), (scaled_stop_x-1, scaled_stop_y-1), self.palette[level], 2 ** level)

        cv2.imwrite(save_path, resized_img)

    @staticmethod
    def save_hdf5(output_path, asset_dict, attr_dict, mode='a'):
        file = h5py.File(output_path, mode)
        for key, val in asset_dict.items():
            data_shape = val.shape
            if key not in file:
                data_type = val.dtype
                chunk_shape = (1,) + data_shape[1:]
                maxshape = (None,) + data_shape[1:]
                dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape,
                                           dtype=data_type)
                dset[:] = val
            else:
                dset = file[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                dset[-data_shape[0]:] = val
        for key, val in attr_dict.items():
            file.attrs[key] = val
        file.close()
        return output_path

    def multi_level_segment(self):
        h5_path = os.path.join(self.dst, 'coordinates', f'{self.wsi_name}.h5')
        if os.path.exists(h5_path) and self.skip:
            print(f'\n{self.wsi_name} already processed. Skipping...')
            return
        os.makedirs(os.path.dirname(h5_path), exist_ok=True)

        # load the WSI
        print(f'loading {self.wsi_name}...')
        start = time.time()
        img = np.array(self.wsi.read_region((0, 0), self.base_level, self.base_dimensions))
        print(f'WSI loaded in {time.time() - start:.2f}s')
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # following CLAM
        if self.style == 'CLAM':
            img_med = cv2.medianBlur(img_hsv[:, :, 1], self.mthresh)

            # thresholding
            if self.use_otsu:
                print('Using Otsu thresholding')
                _, img_otsu = cv2.threshold(img_med, self.sthresh, self.sthresh_up, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                _, img_otsu = cv2.threshold(img_med, self.sthresh, self.sthresh_up, cv2.THRESH_BINARY)

            # the minimum bounding box of the whole tissue
            contours, _ = cv2.findContours(img_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = np.concatenate(contours)
            # scale the coord to level 0
            scale = self.level_downsamples[self.base_level]
            contours = (contours * scale).astype(np.int32)

        elif self.style == 'DTFD':
            # DTFD's way of preprocessing
            h, s, v = cv2.split(img_hsv)

            hthresh = threshold_otsu(h)
            sthresh = threshold_otsu(s)
            vthresh = threshold_otsu(v)

            minhsv = np.array([hthresh, sthresh, 70], np.uint8)
            maxhsv = np.array([180, 255, vthresh], np.uint8)
            thresh = [minhsv, maxhsv]
            mask = cv2.inRange(img_hsv, thresh[0], thresh[1])

            close_kernel = np.ones((100, 100), dtype=np.uint8)
            image_close_img = Image.fromarray(cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel))
            open_kernel = np.ones((60, 60), dtype=np.uint8)
            image_open_np = cv2.morphologyEx(np.array(image_close_img), cv2.MORPH_OPEN, open_kernel)

            contours, _ = cv2.findContours(image_open_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   #_, contours, _ = cv2.findContours(image_open_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = np.concatenate(contours)
            scale = self.level_downsamples[self.base_level]
            contours = (contours * scale).astype(np.int32)

        else:
            raise ValueError(f'Unknown style: {self.style}')

        x, y, w, h = cv2.boundingRect(contours)

        img_w, img_h = self.level_dimensions[0]
        base_patch_size = self.patch_size * scale[0]

        if self.padding:
            stop_y = y + h
            stop_x = x + w
        else:
            # drop the last patch if it is smaller than the patch size
            stop_y = min(y + h, img_h - base_patch_size + 1)
            stop_x = min(x + w, img_w - base_patch_size + 1)

        print("Bounding box: ", x, y, w, h)
        print("Contour area: ", cv2.contourArea(contours))

        # No need to check the holes. Directly generate the mesh
        asset_dict = {}
        for i in range(self.num_levels):
            factor = factor * self.downsample_factor[i - 1] if i > 0 else 1
            step_size = int(base_patch_size * factor)
            x_range = np.arange(x, stop_x, step_size)
            y_range = np.arange(y, stop_y, step_size)
            x_coord, y_coord = np.meshgrid(x_range, y_range, indexing='ij')
            asset_dict[f'level_{i}'] = np.stack([x_coord, y_coord], axis=-1)
        
        # For faster downstream feature extraction, directly resize and save the patches
        # use the same reading method as in the feature extraction to ensure it is correct
        if self.save_patch:
            patch_path = os.path.join(self.dst, 'patches', f'{self.wsi_name}')
            os.makedirs(patch_path, exist_ok=True)
            for i in range(self.num_levels):
                level_save_path = os.path.join(patch_path, f'level_{i}')
                os.makedirs(level_save_path, exist_ok=True)
                level_coords = asset_dict[f'level_{i}']
                factor = factor * self.downsample_factor[i - 1] if i > 0 else 1
                level_patch_size = int(self.patch_size * factor)
                
                for m in range(level_coords.shape[0]):
                    for n in range(level_coords.shape[1]):
                        x, y = level_coords[m, n]
                        patch = np.array(self.wsi.read_region((int(x), int(y)), self.base_level, (level_patch_size, level_patch_size)))
                        cv2.imwrite(os.path.join(level_save_path, f'{m}_{n}_.png'), patch)
        
            overview = cv2.resize(img, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(patch_path, 'overview.png'), overview)

        if self.visualize:
            self._visualize_grid(img, asset_dict, stop_x, stop_y)

        attr_dict = {'base_level': self.base_level, 'base_dimensions': self.base_dimensions,
                     'base_downsample': self.base_downsample, 'padding': self.padding,
                     'downsample_factor': self.downsample_factor, 'patch_size': self.patch_size,
                     'num_levels': self.num_levels}

        assert asset_dict, "Asset dictionary is empty"

        self.save_hdf5(h5_path, asset_dict, attr_dict, mode='w')
