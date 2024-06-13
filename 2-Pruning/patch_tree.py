import os
import h5py
import pathlib
import shutil
from PIL import Image
import numpy as np
from anytree import AnyNode, LevelOrderIter
from torchvision import transforms
from torch.utils.data import IterableDataset, Dataset


class MyNode(AnyNode):
    
    def delete(self):
        # delete the patch file of the current node and its children
        os.remove(self.patch_path)
        self.patch_path = ''
        self.parent = None
        for child in self.children:
            child.delete()
        del self
        

class PatchTree:

    def __init__(self, coord_path, patch_root, lowest_level=0):

        self.patch_root = patch_root
        self.coord_path = coord_path
        self.lowest_level = lowest_level

        self.shapes = {}
        with h5py.File(self.coord_path, "r") as f:
            self.num_levels = f.attrs['num_levels']
            self.patch_size = f.attrs['patch_size']
            self.downsample_factor = f.attrs['downsample_factor']
            for i in range(self.num_levels):
                self.shapes[f'level_{i}'] = f[f'level_{i}'].shape
        
        self.patch_paths = {}
        for i in range(self.num_levels):
            level_shape = self.shapes[f'level_{i}']
            self.patch_paths[f'level_{i}'] = np.full([level_shape[0], level_shape[1]], '', dtype=object)
            for x in range(level_shape[0]):
                for y in range(level_shape[1]):
                    patch_path = os.path.join(self.patch_root, f'level_{i}', f'{x}_{y}_.png')
                    if os.path.exists(patch_path):
                        self.patch_paths[f'level_{i}'][x, y] = patch_path
        
        self.root = MyNode(parent=None, patch_path='', level=self.num_levels, i=0, j=0)
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
        max_child_i = min((cur_i + 1) * self.downsample_factor, self.patch_paths[f'level_{child_level}'].shape[0])
        max_child_j = min((cur_j + 1) * self.downsample_factor, self.patch_paths[f'level_{child_level}'].shape[1])
        # if current node is the pseudo root, include all the nodes in the next level as the children
        if cur_level == self.num_levels:
            max_child_i = self.patch_paths[f'level_{child_level}'].shape[0]
            max_child_j = self.patch_paths[f'level_{child_level}'].shape[1]

        range_i = list(range(min_child_i, max_child_i))
        range_j = list(range(min_child_j, max_child_j))

        # if the img is the parent of leaves, dirrectly add the leaves as children and terminate recursion
        if cur_level == self.lowest_level + 1:
            for child_j in range_j:
                for child_i in range_i:
                    temp_path = self.patch_paths[f'level_{child_level}'][child_i, child_j]
                    if temp_path == '':
                        continue
                    temp = MyNode(parent=cur_node, i=child_i, j=child_j, level=child_level, patch_path=temp_path)
            return 

        # if the img is not the parent of leaves, recursively add the children
        else:
            for child_j in range_j:
                for child_i in range_i:
                    temp_path = self.patch_paths[f'level_{child_level}'][child_i, child_j]
                    if temp_path == '':
                        continue
                    temp = MyNode(parent=cur_node, i=child_i, j=child_j, level=child_level, patch_path=temp_path)
                    self._recursive_scan(temp)


class LevelPatchDataset(Dataset):

    def __init__(self, tree: MyNode, level: int):
        self.tree = tree
        self.level = level
        self.node_list = [node for node in LevelOrderIter(tree.root) if node.level == level]
    
    def __len__(self):
        return len(self.node_list)
    
    def __getitem__(self, idx):
        node = self.node_list[idx]
        filename = pathlib.Path(node.patch_path).name
        patch = Image.open(node.patch_path).convert('RGB')
        patch = self.transform(patch)

        return patch, filename, idx

    @staticmethod
    def transform(images):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        return transform(images)


