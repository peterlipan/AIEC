import os
import torch
import random
import numpy as np
from itertools import product
from collections import defaultdict
from anytree import AnyNode, PreOrderIter, LevelOrderIter, RenderTree, PostOrderIter


class MyNode(AnyNode):
    def drop(self):
        # remove the current node and its children from the tree
        self.parent = None
        for child in self.children:
            child.drop()
        del self


class AbstractScan(object):
    def __init__(self, num_levels, downsample_factor, lowest_level=0, p=1, p_i=.5, p_j=.5, mode='features'):
        self.num_levels = num_levels
        self.downsample_factor = downsample_factor
        self.lowest_level = lowest_level
        self.p = p
        self.p_i = p_i
        self.p_j = p_j
        self.reverse_i = False
        self.reverse_j = False
        self.mode = mode
    
    def _recursive_scan(self, cur_node: MyNode):
        raise NotImplementedError("Subclasses should implement this!")
    
    def _is_valid(self, sample):
        if self.mode == 'features':
            return torch.any(sample != 0)
        elif self.mode == 'coordinates':
            return torch.any(sample != -1)
        elif self.mode == 'patches':
            return os.path.exists(sample)
        else:
            raise ValueError('Invalid mode')
    
    def _get_child_region(self, cur_level, cur_i, cur_j):
        # spatial information of the children
        child_level = cur_level - 1
        if cur_level == self.num_levels:
            # if current node is the pseudo root, include all the nodes in the next level as the children
            min_child_i = min_child_j = 0
            max_child_i = self.data[f'level_{child_level}'].shape[0]
            max_child_j = self.data[f'level_{child_level}'].shape[1]
        else:  
            factor = self.downsample_factor[child_level]
            min_child_i = cur_i * factor
            min_child_j = cur_j * factor
            max_child_i = min((cur_i + 1) * factor, self.data[f'level_{child_level}'].shape[0])
            max_child_j = min((cur_j + 1) * factor, self.data[f'level_{child_level}'].shape[1])
        range_i = list(range(min_child_i, max_child_i))
        range_j = list(range(min_child_j, max_child_j))
        return range_i, range_j
    
    def __call__(self, data):
        if random.random() < self.p_i:
            self.reverse_i = True
        if random.random() < self.p_j:
            self.reverse_j = True

        self.data = data
        root = MyNode(i=0, j=0, level=self.num_levels, data=data['overview'][0,0])
        self._recursive_scan(root)
        return root


class HorizontalRasterScan(AbstractScan):

    # recursively build tree in the horizontal direction of each level
    def _recursive_scan(self, cur_node: MyNode):
        # spatial information of the current node
        cur_level = cur_node.level
        child_level = cur_level - 1
        cur_i = cur_node.i
        cur_j = cur_node.j

        range_i, range_j = self._get_child_region(cur_level, cur_i, cur_j)

        if self.reverse_i:
            range_i = range_i[::-1]
        if self.reverse_j:
            range_j = range_j[::-1]

        # Horizontal scan
        for child_i in range_i:
            for child_j in range_j:
                temp_data = self.data[f'level_{child_level}'][child_i, child_j]
                if not self._is_valid(temp_data):
                    continue
                temp = MyNode(parent=cur_node, i=child_i, j=child_j, level=child_level, data=temp_data)
                # if the child is not the leaf node, recursively add the children
                if child_level > self.lowest_level:
                    self._recursive_scan(temp)
        return 


class VerticalRasterScan(AbstractScan):

    # recursively build tree in the vertical direction of each level
    def _recursive_scan(self, cur_node: MyNode):
        # spatial information of the current node
        cur_level = cur_node.level
        child_level = cur_level - 1
        cur_i = cur_node.i
        cur_j = cur_node.j

        range_i, range_j = self._get_child_region(cur_level, cur_i, cur_j)

        if self.reverse_i:
            range_i = range_i[::-1]
        if self.reverse_j:
            range_j = range_j[::-1]

        # Vertical scan
        for child_j in range_j:
            for child_i in range_i:
                temp_data = self.data[f'level_{child_level}'][child_i, child_j]
                if not self._is_valid(temp_data):
                    continue
                temp = MyNode(parent=cur_node, i=child_i, j=child_j, level=child_level, data=temp_data)
                # if the child is not the leaf node, recursively add the children
                if child_level > self.lowest_level:
                    self._recursive_scan(temp)
        return 


class HorizontalZigzagScan(AbstractScan):
        
    # recursively build tree in the horizontal direction of each level
    def _recursive_scan(self, cur_node: MyNode):
        # spatial information of the current node
        cur_level = cur_node.level
        child_level = cur_level - 1
        cur_i = cur_node.i
        cur_j = cur_node.j
    
        # spatial information of the children
        range_i, range_j = self._get_child_region(cur_level, cur_i, cur_j)

        if self.reverse_i:
            range_i = range_i[::-1]
        if self.reverse_j:
            range_j = range_j[::-1]
        

        # Horizontal zigzag scan
        for count_i, child_i in enumerate(range_i):
            if count_i % 2 == 0:
                for child_j in range_j:
                    temp_data = self.data[f'level_{child_level}'][child_i, child_j]
                    if not self._is_valid(temp_data):
                        continue
                    temp = MyNode(parent=cur_node, i=child_i, j=child_j, level=child_level, data=temp_data)
                    # if the child is not the leaf node, recursively add the children
                    if child_level > self.lowest_level:
                        self._recursive_scan(temp)
            else:
                for child_j in reversed(range_j):
                    temp_data = self.data[f'level_{child_level}'][child_i, child_j]
                    if not self._is_valid(temp_data):
                        continue
                    temp = MyNode(parent=cur_node, i=child_i, j=child_j, level=child_level, data=temp_data)
                    # if the child is not the leaf node, recursively add the children
                    if child_level > self.lowest_level:
                        self._recursive_scan(temp)
        return


class VerticalZigzagScan(AbstractScan):

    # recursively build tree in the vertical direction of each level
    def _recursive_scan(self, cur_node: MyNode):
        # spatial information of the current node
        cur_level = cur_node.level
        child_level = cur_level - 1
        cur_i = cur_node.i
        cur_j = cur_node.j
    
       # spatial information of the children
        range_i, range_j = self._get_child_region(cur_level, cur_i, cur_j)

        if self.reverse_i:
            range_i = range_i[::-1]
        if self.reverse_j:
            range_j = range_j[::-1]
        

        # Vertical zigzag scan
        for count_j, child_j in enumerate(range_j):
            if count_j % 2 == 0:
                for child_i in range_i:
                    temp_data = self.data[f'level_{child_level}'][child_i, child_j]
                    if not self._is_valid(temp_data):
                        continue
                    temp = MyNode(parent=cur_node, i=child_i, j=child_j, level=child_level, data=temp_data)
                    # if the child is not the leaf node, recursively add the children
                    if child_level > self.lowest_level:
                        self._recursive_scan(temp)
            else:
                for child_i in reversed(range_i):
                    temp_data = self.data[f'level_{child_level}'][child_i, child_j]
                    if not self._is_valid(temp_data):
                        continue
                    temp = MyNode(parent=cur_node, i=child_i, j=child_j, level=child_level, data=temp_data)
                    # if the child is not the leaf node, recursively add the children
                    if child_level > self.lowest_level:
                        self._recursive_scan(temp)
        return


class DiagonalScan(AbstractScan):
    """Traverses children diagonally (top-left to bottom-right)"""
    def _recursive_scan(self, cur_node: MyNode):
        cur_level = cur_node.level
        child_level = cur_level - 1
        cur_i = cur_node.i
        cur_j = cur_node.j
        
        range_i, range_j = self._get_child_region(cur_level, cur_i, cur_j)
        
        if self.reverse_i:
            range_i = range_i[::-1]
        if self.reverse_j:
            range_j = range_j[::-1]
        
        # Get all indices and sort by diagonal sum (i+j)
        indices = [(i, j) for i in range_i for j in range_j]
        indices.sort(key=lambda x: (x[0] + x[1], x[0]))
        
        for child_i, child_j in indices:
            temp_data = self.data[f'level_{child_level}'][child_i, child_j]
            if not self._is_valid(temp_data):
                continue
            temp = MyNode(parent=cur_node, i=child_i, j=child_j, level=child_level, data=temp_data)
            if child_level > self.lowest_level:
                self._recursive_scan(temp)


class SpiralScan(AbstractScan):
    """Traverses from center outward in spiral pattern"""
    def _recursive_scan(self, cur_node: MyNode):
        cur_level = cur_node.level
        child_level = cur_level - 1
        cur_i = cur_node.i
        cur_j = cur_node.j
        
        range_i, range_j = self._get_child_region(cur_level, cur_i, cur_j)
        
        if self.reverse_i:
            range_i = range_i[::-1]
        if self.reverse_j:
            range_j = range_j[::-1]
        
        # Generate spiral indices
        indices = []
        top, bottom = 0, len(range_i)-1
        left, right = 0, len(range_j)-1
        
        while top <= bottom and left <= right:
            # Rightward
            for j in range(left, right+1):
                indices.append((range_i[top], range_j[j]))
            top += 1
            
            # Downward
            for i in range(top, bottom+1):
                indices.append((range_i[i], range_j[right]))
            right -= 1
            
            # Leftward (if applicable)
            if top <= bottom:
                for j in range(right, left-1, -1):
                    indices.append((range_i[bottom], range_j[j]))
                bottom -= 1
            
            # Upward (if applicable)
            if left <= right:
                for i in range(bottom, top-1, -1):
                    indices.append((range_i[i], range_j[left]))
                left += 1
        
        # Process in spiral order
        for child_i, child_j in indices:
            temp_data = self.data[f'level_{child_level}'][child_i, child_j]
            if not self._is_valid(temp_data):
                continue
            temp = MyNode(parent=cur_node, i=child_i, j=child_j, level=child_level, data=temp_data)
            if child_level > self.lowest_level:
                self._recursive_scan(temp)


class AbstractReadout(object):
    def __init__(self, p=1, levels=None):
        self.p = p
        self.levels = levels
    
    def _readout_func(self, root):
        raise NotImplementedError("Subclasses should implement this!")
    
    def __call__(self, root):
        return self._readout_func(root)


class DepthFirstReadout(AbstractReadout):
    
    def _readout_func(self, root):            
        return torch.stack([node.data for node in PreOrderIter(root) if node.level in self.levels])


class BreadthFirstReadout(AbstractReadout):
    
    def _readout_func(self, root):
        return torch.stack([node.data for node in LevelOrderIter(root) if node.level in self.levels])


class UpwardsReadout(AbstractReadout):
    
    def _readout_func(self, root):
        return torch.stack([node.data for node in PostOrderIter(root) if node.level in self.levels])
                

class RandomFeatureAugmentation(object):
    def __init__(self, p=1):
        self.p = p
    
    def augment(self, x):
        raise NotImplementedError("Subclasses should implement this!")

    def __call__(self, x):
        if random.random() < self.p:
            return self.augment(x)
        return x


class RandomFeatureJitter(RandomFeatureAugmentation):
    def __init__(self, p=1, std=0.1):
        super().__init__(p)
        self.std = std

    def augment(self, x):
        noise = torch.randn_like(x) * self.std
        return x + noise


class RandomGaussianNoise(RandomFeatureAugmentation):
    def __init__(self, p=1, mean=0.0, std=0.1):
        super().__init__(p)
        self.mean = mean
        self.std = std

    def augment(self, x):
        noise = torch.randn_like(x) * self.std + self.mean
        return x + noise


class RandomFeatureScaling(RandomFeatureAugmentation):
    def __init__(self, p=1, scale_range=(0.5, 1.5)):
        super().__init__(p)
        self.scale_range = scale_range

    def augment(self, x):
        scale = random.uniform(*self.scale_range)
        return x * scale


class RandomFeatureClipping(RandomFeatureAugmentation):
    def __init__(self, p=1, clip_range=(0.0, 1.0)):
        super().__init__(p)
        self.clip_range = clip_range

    def augment(self, x):
        return torch.clamp(x, *self.clip_range)


class RandomFeatureDropout(RandomFeatureAugmentation):
    def __init__(self, p=1, dropout_rate=0.1):
        super().__init__(p)
        self.dropout_rate = dropout_rate

    def augment(self, x):
        mask = (torch.rand_like(x) > self.dropout_rate).float()
        return x * mask


class RandomFeatureShift(RandomFeatureAugmentation):
    def __init__(self, p=1, shift_std=0.1):
        super().__init__(p)
        self.shift_std = shift_std

    def augment(self, x):
        shift = torch.randn(1, x.shape[-1], device=x.device) * self.shift_std
        return x + shift


class OneOf(object):
    def __init__(self, transforms: list, p: float = 1):
        self.transforms = transforms
        transforms_ps = [t.p for t in self.transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, x):
        transform = np.random.choice(self.transforms, p=self.transforms_ps)
        return transform(x)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class TreeDropOut:
    def __init__(self, num_levels: int, probs: list, keep_root: bool = True, p: float = 0.5):
        assert len(probs) == num_levels, 'The number of probabilities should match the number of levels'
        self.num_levels = num_levels
        self.probs = probs
        self.keep_root = keep_root
        self.p = p

    def __call__(self, root):
        for i in reversed(range(self.num_levels)):
            nodes = [node for node in LevelOrderIter(root) if node.level == i]
            for node in nodes:
                if self.keep_root and node.is_root:
                    continue
                # Skip if parent already dropped (i.e., has no parent)
                if node.is_root or node.parent is None:
                    continue
                if random.random() < self.probs[i]:
                    node.drop()
        return root


class TreeCut:
    def __init__(self, p=0.5, max_depth=2):
        self.p = p
        self.max_depth = max_depth

    def __call__(self, root):
        for node in PreOrderIter(root):
            if node.depth <= self.max_depth and not node.is_root:
                if random.random() < self.p:
                    node.children = []
        return root


class TreeShuffle:
    def __init__(self, p=0.5, levels=None):
        self.p = p
        self.levels = levels  # List of levels to shuffle; None = all

    def __call__(self, root):
        if random.random() >= self.p:
            return root

        level_to_nodes = defaultdict(list)
        for node in LevelOrderIter(root):
            if self.levels is None or node.level in self.levels:
                level_to_nodes[node.level].append(node)

        for level_nodes in level_to_nodes.values():
            data_list = [n.data.clone() for n in level_nodes]
            random.shuffle(data_list)
            for node, new_data in zip(level_nodes, data_list):
                node.data = new_data

        return root


class TreeNodeDrop:
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, root):
        for node in PreOrderIter(root):
            if not node.is_root and random.random() < self.p:
                node.data = torch.zeros_like(node.data)
        return root


class TreeMergeLeaves:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, root):
        for node in PreOrderIter(root):
            leaves = [child for child in node.children if child.is_leaf]
            if len(leaves) >= 2 and random.random() < self.p:
                merged_data = torch.stack([leaf.data for leaf in leaves]).mean(0)
                node.children = [n for n in node.children if n not in leaves]
                new_leaf = MyNode(name="merged_leaf", parent=node, data=merged_data, level=node.level - 1)
        return root


def get_train_transforms(n_experts, num_levels, downsample_factor, lowest_level, dropout, visible_levels):
    scan_augmentations = [
        HorizontalRasterScan(num_levels, downsample_factor, lowest_level),
        VerticalRasterScan(num_levels, downsample_factor, lowest_level),
        HorizontalZigzagScan(num_levels, downsample_factor, lowest_level),
        VerticalZigzagScan(num_levels, downsample_factor, lowest_level),
        # DiagonalScan(num_levels, downsample_factor, lowest_level),
        # SpiralScan(num_levels, downsample_factor, lowest_level)
    ]
    readout_augmentations = [
        DepthFirstReadout(levels=visible_levels),
        BreadthFirstReadout(levels=visible_levels),
        # UpwardsReadout(levels=visible_levels)
    ]
    tree_augmentations = Compose([
        OneOf([
            TreeDropOut(num_levels, dropout, keep_root=True, p=0.5),
            TreeCut(p=0.5, max_depth=2),
        ]),
        TreeShuffle(p=0.5, levels=visible_levels),
        TreeNodeDrop(p=0.3),
        TreeMergeLeaves(p=0.2)
    ])
    feature_augmentations = Compose([
        OneOf([
            RandomFeatureJitter(p=0.5, std=0.1),
            RandomGaussianNoise(p=0.5, mean=0.1, std=0.1),
        ]),
        OneOf([
            RandomFeatureScaling(p=0.5, scale_range=(0.5, 1.5)),
            RandomFeatureClipping(p=0.5, clip_range=(0.0, 1.0)),
        ]),
        RandomFeatureDropout(p=0.5, dropout_rate=0.1),
        RandomFeatureShift(p=0.5, shift_std=0.1)
    ])

    transforms  = []
    all_pairs = list(product(scan_augmentations, readout_augmentations))
    for i in range(n_experts):
        # Fix the scan and readout for each expert to keep the topology consistent
        scan, readout = all_pairs[i % len(all_pairs)]
        transforms.append(
            Compose([
                scan,
                tree_augmentations,
                readout,
                feature_augmentations
            ])
        )
    return transforms
    

def get_test_transforms(n_experts, num_levels, downsample_factor, lowest_level, visible_levels):
    scans = [
        HorizontalRasterScan(num_levels, downsample_factor, lowest_level, p_i=0, p_j=0),
        VerticalRasterScan(num_levels, downsample_factor, lowest_level, p_i=0, p_j=0),
        HorizontalZigzagScan(num_levels, downsample_factor, lowest_level, p_i=0, p_j=0),
        VerticalZigzagScan(num_levels, downsample_factor, lowest_level, p_i=0, p_j=0),
        # DiagonalScan(num_levels, downsample_factor, lowest_level, p_i=0, p_j=0),
        # SpiralScan(num_levels, downsample_factor, lowest_level, p_i=0, p_j=0)
    ]
    readouts = [
        DepthFirstReadout(levels=visible_levels),
        BreadthFirstReadout(levels=visible_levels),
        # UpwardsReadout(levels=visible_levels)
    ]
    transforms = []
    all_pairs = list(product(scans, readouts))
    for i in range(n_experts):
        # Fix the scan and readout for each expert to keep the topology consistent
        scan, readout = all_pairs[i % len(all_pairs)]
        transforms.append(
            Compose([
                scan,
                readout
            ])
        )
    return transforms


if __name__ == '__main__':
    # Test the scans
    num_levels = 3
    downsample_factor = 3
    data = {}
    shapes = [(32, 78), (8, 20), (2, 5)]

    for i in range(num_levels):
        data[f'level_{i}'] = torch.randn(shapes[i])
        print(f'level_{i} ', f'shape: {shapes[i]}')
    data['overview'] = torch.randn((1,1))
    
    scan = HorizontalRasterScan(num_levels=3, downsample_factor=[3, 3], lowest_level=0, p_i=0.5, p_j=0.5)
    drop = TreeDropOut(3, [0.1, 0.2, 0.3])
    readout = BreadthFirstReadout()
    root = scan(data)
    root = drop(root)
    # Depth first traversal
    for node in PreOrderIter(root):
        print(f'level{node.level}', node.i, node.j)
    
    print('---------------------------------')
    print(RenderTree(root))
    print('---------------------------------')
    print(readout(root).shape)
    print(scan.reverse_i, scan.reverse_j)



