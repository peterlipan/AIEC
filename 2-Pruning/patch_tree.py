import os
import cv2
import copy
import h5py
import openslide
from PIL import Image
import numpy as np
from pathlib import Path
from anytree import AnyNode, LevelOrderIter
from torchvision import transforms
from torch.utils.data import IterableDataset, Dataset


class MyNode(AnyNode):
    def delete(self):
        # delete the patch file of the current node and its children
        if isinstance(self.data, str):
            os.remove(self.data)
        self.data = None
        # detach
        self.parent = None
        for child in self.children:
            child.delete()
        del self
        

class PatchTree:

    def __init__(self, coord_path=None, patch_root=None, wsi_path=None, save_path=None, lowest_level=0, mode='patch'):

        self.patch_root = patch_root
        self.coord_path = coord_path
        self.lowest_level = lowest_level
        self.mode = mode
        self.attributes = {}

        self.save_path = save_path

        self.shapes = {}
        self.tree_data = {}
        if mode == 'coordinate':
            self.coord2write = {}
            self.wsi = openslide.OpenSlide(wsi_path)
        with h5py.File(self.coord_path, "r") as f:
            for key, val in f.attrs.items():
                self.attributes[key] = val
            self.num_levels = f.attrs['num_levels']
            self.patch_size = f.attrs['patch_size']
            self.downsample_factor = f.attrs['downsample_factor']
            self.base_downsample = f.attrs['base_downsample']
            self.base_level = f.attrs['base_level']
            for i in range(self.num_levels):
                self.shapes[f'level_{i}'] = f[f'level_{i}'].shape
                if mode == 'coordinate':
                    self.coord2write[f'level_{i}'] = -1 * np.ones_like(np.array(f[f'level_{i}']))
                    self.tree_data[f'level_{i}'] = np.array(f[f'level_{i}'])

        if mode == 'patch':
            for i in range(self.num_levels):
                level_shape = self.shapes[f'level_{i}']
                self.tree_data[f'level_{i}'] = np.full([level_shape[0], level_shape[1]], '', dtype=object)
                for x in range(level_shape[0]):
                    for y in range(level_shape[1]):
                        patch_path = os.path.join(self.patch_root, f'level_{i}', f'{x}_{y}_.png')
                        if os.path.exists(patch_path):
                            self.tree_data[f'level_{i}'][x, y] = patch_path
        
        self.root = MyNode(parent=None, data=None, level=self.num_levels, i=0, j=0)
        self._recursive_scan(self.root)

    def _recursive_scan(self, cur_node: MyNode):
        # spatial information of the current node
        cur_level = cur_node.level
        cur_i = cur_node.i
        cur_j = cur_node.j

        # spatial information of the children
        # if current node is the pseudo root, include all the nodes in the next level as the children
        child_level = cur_level - 1
        if cur_level == self.num_levels:
            min_child_i = min_child_j = 0
            max_child_i = self.shapes[f'level_{child_level}'][0]
            max_child_j = self.shapes[f'level_{child_level}'][1]
        else:  
            factor = self.downsample_factor[child_level]
            min_child_i = cur_i * factor
            min_child_j = cur_j * factor
            max_child_i = min((cur_i + 1) * factor, self.shapes[f'level_{child_level}'][0])
            max_child_j = min((cur_j + 1) * factor, self.shapes[f'level_{child_level}'][1])

        range_i = list(range(min_child_i, max_child_i))
        range_j = list(range(min_child_j, max_child_j))


        for child_j in range_j:
            for child_i in range_i:
                temp_data = self.tree_data[f'level_{child_level}'][child_i, child_j]
                if not self._patch_exists(temp_data):
                    continue
                temp = MyNode(parent=cur_node, i=child_i, j=child_j, level=child_level, data=temp_data)
                # if the child is not the leaf node, recursively add the children
                if child_level > self.lowest_level:
                    self._recursive_scan(temp)
        return 

    def _patch_exists(self, data):
        if data is None:
            return False
        if self.mode == 'patch':
            return os.path.exists(data)
        elif self.mode == 'coordinate':
            return not all(np.array(data) == -1 * np.ones_like(data))
        return True


    @staticmethod
    def save_hdf5(output_path, asset_dict, attr_dict, mode='w'):
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

    def save(self):
        for node in LevelOrderIter(self.root):
            if node.data is None:
                continue
            self.coord2write[f'level_{node.level}'][node.i, node.j] = node.data
        self.save_hdf5(self.save_path, self.coord2write, self.attributes, mode='w')
    
    def visualize(self, orgin_coord, vis_path):
        visualize_width = 1024
        palette = [(173, 216, 230, 255), (255, 182, 193, 255), (152, 251, 152, 255), (230, 230, 250, 255),
                        (255, 255, 0, 255), (255, 165, 0, 255), (255, 0, 255, 255), (64, 224, 208, 255),
                        (168, 168, 120, 255), (210, 105, 30, 255), (255, 199, 0, 255), (138, 54, 15, 255)]
        scale = self.wsi.level_downsamples[-1]
        img = np.array(self.wsi.read_region((0, 0), len(self.wsi.level_dimensions) - 1, self.wsi.level_dimensions[-1]).convert('RGB'))

        height, width, _ = img.shape
        new_height = int(visualize_width * height / width)
        resized_img = cv2.resize(img, (visualize_width, new_height), interpolation=cv2.INTER_CUBIC)
        resized_height, resized_width, _ = resized_img.shape

        stop_x = max([orgin_coord[f'level_{level}'][:, :, 0].max() for level in range(self.num_levels)])
        stop_y = max([orgin_coord[f'level_{level}'][:, :, 1].max() for level in range(self.num_levels)])
        scaled_stop_x = int(stop_x / scale / width * resized_width)
        scaled_stop_y = int(stop_y / scale / height * resized_height)
        
        # draw the coords
        for level in range(self.num_levels):
            grid_x, grid_y = orgin_coord[f'level_{level}'][:, :, 0], orgin_coord[f'level_{level}'][:, :, 1]
            scaled_grid_x = grid_x[:, 0] / scale / width * resized_width
            scaled_grid_y = grid_y[0, :] / scale / height * resized_height

            scaled_start_x = int(min(scaled_grid_x))
            scaled_start_y = int(min(scaled_grid_y))

            for x in set(scaled_grid_x):
                cv2.line(resized_img, (int(x), scaled_start_y), (int(x), scaled_stop_y - 1), palette[level], 2 ** level)
            
            for y in set(scaled_grid_y):
                cv2.line(resized_img, (scaled_start_x, int(y)), (scaled_stop_x - 1, int(y)), palette[level], 2 ** level)
            
            cv2.line(resized_img, (scaled_stop_x-1, scaled_start_y), (scaled_stop_x-1, scaled_stop_y-1), palette[level], 2 ** level)
            cv2.line(resized_img, (scaled_start_x, scaled_stop_y-1), (scaled_stop_x-1, scaled_stop_y-1), palette[level], 2 ** level)
        
        # visulized the pruned regions as black
        for level in range(self.num_levels):
            factor = factor * self.downsample_factor[level - 1] if level > 0 else 1
            for i in range(self.shapes[f'level_{level}'][0]):
                for j in range(self.shapes[f'level_{level}'][1]):
                    if all(self.coord2write[f'level_{level}'][i, j] == -1):
                        x = orgin_coord[f'level_{level}'][i,j][0] / scale / width * resized_width
                        y = orgin_coord[f'level_{level}'][i,j][1] / scale / height * resized_height
                        level_patch_size = self.patch_size * self.base_downsample * factor
                        scaled_patch_size = level_patch_size / scale / width * resized_width
                        cv2.rectangle(resized_img, (int(x), int(y)), (int(x + scaled_patch_size), int(y + scaled_patch_size)), (0, 0, 0), -1)

        cv2.imwrite(vis_path, resized_img)


class LevelPatchDataset(Dataset):

    def __init__(self, tree: MyNode, level: int, patch_size: int, mode: str = 'patch'):
        self.tree = tree
        self.level = level
        self.patch_size = patch_size
        self.mode = mode
        self.node_list = [node for node in LevelOrderIter(tree.root) if node.level == level]
        self.level_patch_size = int(patch_size * np.prod(self.tree.downsample_factor[:level]))
    
    def __len__(self):
        return len(self.node_list)
    
    def __getitem__(self, idx):
        node = self.node_list[idx]
        filename = f"{node.i}_{node.j}_.png"
        if self.mode == 'patch':
            patch = Image.open(node.data).convert('RGB')
        elif self.mode == 'coordinate':
            patch = self.tree.wsi.read_region(node.data, self.tree.base_level, (self.level_patch_size, self.level_patch_size)).convert('RGB')
        patch = self.transform(patch)

        return patch, filename, idx

    def transform(self, images):
        transform = transforms.Compose([
            transforms.Resize((self.patch_size, self.patch_size)),
            transforms.ToTensor()
        ])
        return transform(images)


