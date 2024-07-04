import os
import h5py
import openslide
from PIL import Image
import numpy as np
from anytree import AnyNode, LevelOrderIter
from torchvision import transforms
from torch.utils.data import IterableDataset, Dataset


class MyNode(AnyNode):
    def delete(self):
        # delete the patch file of the current node and its children
        self.data = None
        # detach
        self.parent = None
        for child in self.children:
            child.delete()
        del self
        

class PatchTree:

    def __init__(self, coord_path=None, patch_root=None, wsi_path=None, lowest_level=0, mode='patch'):

        self.patch_root = patch_root
        self.coord_path = coord_path
        self.lowest_level = lowest_level
        self.mode = mode

        self.shapes = {}
        self.tree_data = {}
        if mode == 'coordinate':
            self.coord2write = {}
            self.wsi = openslide.OpenSlide(wsi_path)
        with h5py.File(self.coord_path, "r") as f:
            self.attributes = f.attrs
            self.num_levels = f.attrs['num_levels']
            self.patch_size = f.attrs['patch_size']
            self.downsample_factor = f.attrs['downsample_factor']
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
        child_level = cur_level - 1
        min_child_i = cur_i * self.downsample_factor
        min_child_j = cur_j * self.downsample_factor
        max_child_i = min((cur_i + 1) * self.downsample_factor, self.shapes[f'level_{child_level}'][0])
        max_child_j = min((cur_j + 1) * self.downsample_factor, self.shapes[f'level_{child_level}'][1])
        # if current node is the pseudo root, include all the nodes in the next level as the children
        if cur_level == self.num_levels:
            max_child_i = self.shapes[f'level_{child_level}'][0]
            max_child_j = self.shapes[f'level_{child_level}'][1]

        range_i = list(range(min_child_i, max_child_i))
        range_j = list(range(min_child_j, max_child_j))

        # if the img is the parent of leaves, dirrectly add the leaves as children and terminate recursion
        if cur_level == self.lowest_level + 1:
            for child_j in range_j:
                for child_i in range_i:
                    temp_data = self.tree_data[f'level_{child_level}'][child_i, child_j]
                    if temp_data is None:
                        continue
                    temp = MyNode(parent=cur_node, i=child_i, j=child_j, level=child_level, data=temp_data)
            return 

        # if the img is not the parent of leaves, recursively add the children
        else:
            for child_j in range_j:
                for child_i in range_i:
                    temp_data = self.tree_data[f'level_{child_level}'][child_i, child_j]
                    if temp_data is None:
                        continue
                    temp = MyNode(parent=cur_node, i=child_i, j=child_j, level=child_level, data=temp_data)
                    self._recursive_scan(temp)

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

    def save_changes(self):
        for node in LevelOrderIter(self.root):
            if node.data is None:
                continue
            self.coord2write[f'level_{node.level}'][node.i, node.j] = node.data
        self.save_hdf5(self.coord_path, self.coord2write, self.attributes, mode='w')


class LevelPatchDataset(Dataset):

    def __init__(self, tree: MyNode, level: int, patch_size: int, mode: str = 'patch'):
        self.tree = tree
        self.level = level
        self.patch_size = patch_size
        self.mode = mode
        self.node_list = [node for node in LevelOrderIter(tree.root) if node.level == level]
        self.level_patch_size = patch_size * self.tree.downsample_factor ** level
    
    def __len__(self):
        return len(self.node_list)
    
    def __getitem__(self, idx):
        node = self.node_list[idx]
        filename = f"{node.i}_{node.j}_.png"
        if self.mode == 'patch':
            patch = Image.open(node.data).convert('RGB')
            patch = self.transform(patch)
        elif self.mode == 'coordinate':
            patch = self.tree.wsi.read_region((node.i, node.j), self.tree.base_level, (self.level_patch_size, self.level_patch_size))

        return patch, filename, idx

    def transform(self, images):
        transform = transforms.Compose([
            transforms.Resize((self.patch_size, self.patch_size)),
            transforms.ToTensor()
        ])
        return transform(images)


