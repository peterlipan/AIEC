import os
import torch
import random
import numpy as np
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
        cur_i = cur_node.i
        cur_j = cur_node.j

        # spatial information of the children
        child_level = cur_level - 1
        min_child_i = cur_i * self.downsample_factor
        min_child_j = cur_j * self.downsample_factor
        max_child_i = min((cur_i + 1) * self.downsample_factor, self.data[f'level_{child_level}'].shape[0])
        max_child_j = min((cur_j + 1) * self.downsample_factor, self.data[f'level_{child_level}'].shape[1])
        # if current node is the pseudo root, include all the nodes in the next level as the children
        if cur_level == self.num_levels:
            max_child_i = self.data[f'level_{child_level}'].shape[0]
            max_child_j = self.data[f'level_{child_level}'].shape[1]

        range_i = list(range(min_child_i, max_child_i))
        range_j = list(range(min_child_j, max_child_j))
        if self.reverse_i:
            range_i = range_i[::-1]
        if self.reverse_j:
            range_j = range_j[::-1]

        # if the img is the parent of leaves, dirrectly add the leaves as children and terminate recursion
        if cur_level == self.lowest_level + 1:
            # Horizontal scan
            for child_i in range_i:
                for child_j in range_j:
                    temp_data = self.data[f'level_{child_level}'][child_i, child_j]
                    if not self._is_valid(temp_data):
                        continue
                    temp = MyNode(parent=cur_node, i=child_i, j=child_j, level=child_level, data=temp_data)
            return 

        # if the img is not the parent of leaves, recursively add the children
        else:
            # Horizontal scan
            for child_i in range_i:
                for child_j in range_j:
                    temp_data = self.data[f'level_{child_level}'][child_i, child_j]
                    if not self._is_valid(temp_data):
                        continue
                    temp = MyNode(parent=cur_node, i=child_i, j=child_j, level=child_level, data=temp_data)
                    self._recursive_scan(temp)


class VerticalRasterScan(AbstractScan):

    # recursively build tree in the vertical direction of each level
    def _recursive_scan(self, cur_node: MyNode):
        # spatial information of the current node
        cur_level = cur_node.level
        cur_i = cur_node.i
        cur_j = cur_node.j

        # spatial information of the children
        child_level = cur_level - 1
        min_child_i = cur_i * self.downsample_factor
        min_child_j = cur_j * self.downsample_factor
        max_child_i = min((cur_i + 1) * self.downsample_factor, self.data[f'level_{child_level}'].shape[0])
        max_child_j = min((cur_j + 1) * self.downsample_factor, self.data[f'level_{child_level}'].shape[1])
        # if current node is the pseudo root, include all the nodes in the next level as the children
        if cur_level == self.num_levels:
            max_child_i = self.data[f'level_{child_level}'].shape[0]
            max_child_j = self.data[f'level_{child_level}'].shape[1]

        range_i = list(range(min_child_i, max_child_i))
        range_j = list(range(min_child_j, max_child_j))
        if self.reverse_i:
            range_i = range_i[::-1]
        if self.reverse_j:
            range_j = range_j[::-1]

        # if the img is the parent of leaves, dirrectly add the leaves as children and terminate recursion
        if cur_level == self.lowest_level + 1:
            # Vertical scan
            for child_j in range_j:
                for child_i in range_i:
                    temp_data = self.data[f'level_{child_level}'][child_i, child_j]
                    if not self._is_valid(temp_data):
                        continue
                    temp = MyNode(parent=cur_node, i=child_i, j=child_j, level=child_level, data=temp_data)
            return 

        # if the img is not the parent of leaves, recursively add the children
        else:
            # Vertical scan
            for child_j in range_j:
                for child_i in range_i:
                    temp_data = self.data[f'level_{child_level}'][child_i, child_j]
                    if not self._is_valid(temp_data):
                        continue
                    temp = MyNode(parent=cur_node, i=child_i, j=child_j, level=child_level, data=temp_data)
                    self._recursive_scan(temp)


class HorizontalZigzagScan(AbstractScan):
        
    # recursively build tree in the horizontal direction of each level
    def _recursive_scan(self, cur_node: MyNode):
        # spatial information of the current node
        cur_level = cur_node.level
        cur_i = cur_node.i
        cur_j = cur_node.j
    
        # spatial information of the children
        child_level = cur_level - 1
        min_child_i = cur_i * self.downsample_factor
        min_child_j = cur_j * self.downsample_factor
        max_child_i = min((cur_i + 1) * self.downsample_factor, self.data[f'level_{child_level}'].shape[0])
        max_child_j = min((cur_j + 1) * self.downsample_factor, self.data[f'level_{child_level}'].shape[1])
        # if current node is the pseudo root, include all the nodes in the next level as the children
        if cur_level == self.num_levels:
            max_child_i = self.data[f'level_{child_level}'].shape[0]
            max_child_j = self.data[f'level_{child_level}'].shape[1]

        range_i = list(range(min_child_i, max_child_i))
        range_j = list(range(min_child_j, max_child_j))
        if self.reverse_i:
            range_i = range_i[::-1]
        if self.reverse_j:
            range_j = range_j[::-1]
        
        # if the img is the parent of leaves, dirrectly add the leaves as children and terminate recursion
        if cur_level == self.lowest_level + 1:
            # Horizontal zigzag scan
            for count_i, child_i in enumerate(range_i):
                if count_i % 2 == 0:
                    for child_j in range_j:
                        temp_data = self.data[f'level_{child_level}'][child_i, child_j]
                        if not self._is_valid(temp_data):
                            continue
                        temp = MyNode(parent=cur_node, i=child_i, j=child_j, level=child_level, data=temp_data)
                else:
                    for child_j in reversed(range_j):
                        temp_data = self.data[f'level_{child_level}'][child_i, child_j]
                        if not self._is_valid(temp_data):
                            continue
                        temp = MyNode(parent=cur_node, i=child_i, j=child_j, level=child_level, data=temp_data)
            return

        # if the img is not the parent of leaves, recursively add the children
        else:
            # Horizontal zigzag scan
            for count_i, child_i in enumerate(range_i):
                if count_i % 2 == 0:
                    for child_j in range_j:
                        temp_data = self.data[f'level_{child_level}'][child_i, child_j]
                        if not self._is_valid(temp_data):
                            continue
                        temp = MyNode(parent=cur_node, i=child_i, j=child_j, level=child_level, data=temp_data)
                        self._recursive_scan(temp)
                else:
                    for child_j in reversed(range_j):
                        temp_data = self.data[f'level_{child_level}'][child_i, child_j]
                        if not self._is_valid(temp_data):
                            continue
                        temp = MyNode(parent=cur_node, i=child_i, j=child_j, level=child_level, data=temp_data)
                        self._recursive_scan(temp)


class VerticalZigzagScan(AbstractScan):

    # recursively build tree in the vertical direction of each level
    def _recursive_scan(self, cur_node: MyNode):
        # spatial information of the current node
        cur_level = cur_node.level
        cur_i = cur_node.i
        cur_j = cur_node.j
    
        # spatial information of the children
        child_level = cur_level - 1
        min_child_i = cur_i * self.downsample_factor
        min_child_j = cur_j * self.downsample_factor
        max_child_i = min((cur_i + 1) * self.downsample_factor, self.data[f'level_{child_level}'].shape[0])
        max_child_j = min((cur_j + 1) * self.downsample_factor, self.data[f'level_{child_level}'].shape[1])
        # if current node is the pseudo root, include all the nodes in the next level as the children
        if cur_level == self.num_levels:
            max_child_i = self.data[f'level_{child_level}'].shape[0]
            max_child_j = self.data[f'level_{child_level}'].shape[1]

        range_i = list(range(min_child_i, max_child_i))
        range_j = list(range(min_child_j, max_child_j))
        if self.reverse_i:
            range_i = range_i[::-1]
        if self.reverse_j:
            range_j = range_j[::-1]
        
        # if the img is the parent of leaves, dirrectly add the leaves as children and terminate recursion
        if cur_level == self.lowest_level + 1:
            # Vertical zigzag scan
            for count_j, child_j in enumerate(range_j):
                if count_j % 2 == 0:
                    for child_i in range_i:
                        temp_data = self.data[f'level_{child_level}'][child_i, child_j]
                        if not self._is_valid(temp_data):
                            continue
                        temp = MyNode(parent=cur_node, i=child_i, j=child_j, level=child_level, data=temp_data)
                else:
                    for child_i in reversed(range_i):
                        temp_data = self.data[f'level_{child_level}'][child_i, child_j]
                        if not self._is_valid(temp_data):
                            continue
                        temp = MyNode(parent=cur_node, i=child_i, j=child_j, level=child_level, data=temp_data)
            return
    
        # if the img is not the parent of leaves, recursively add the children
        else:
            # Vertical zigzag scan
            for count_j, child_j in enumerate(range_j):
                if count_j % 2 == 0:
                    for child_i in range_i:
                        temp_data = self.data[f'level_{child_level}'][child_i, child_j]
                        if not self._is_valid(temp_data):
                            continue
                        temp = MyNode(parent=cur_node, i=child_i, j=child_j, level=child_level, data=temp_data)
                        self._recursive_scan(temp)
                else:
                    for child_i in reversed(range_i):
                        temp_data = self.data[f'level_{child_level}'][child_i, child_j]
                        if not self._is_valid(temp_data):
                            continue
                        temp = MyNode(parent=cur_node, i=child_i, j=child_j, level=child_level, data=temp_data)
                        self._recursive_scan(temp)


class AbstractReadout(object):
    def __init__(self, p=1, levels=None):
        self.p = p
        self.levels = levels
    
    def _readout_func(self, x):
        raise NotImplementedError("Subclasses should implement this!")
    
    def __call__(self, x):
        return self._readout_func(x)


class DepthFirstReadout(AbstractReadout):
    
    def _readout_func(self, x):
        # drop the pseudo root node
        return torch.stack([x.data for x in PreOrderIter(x) if x.level in self.levels])


class BreadthFirstReadout(AbstractReadout):
    
    def _readout_func(self, x):
        # drop the pseudo root node
        return torch.stack([x.data for x in LevelOrderIter(x) if x.level in self.levels])


class UpwardsReadout(AbstractReadout):
    
    def _readout_func(self, x):
        # drop the pseudo root node
        return torch.stack([x.data for x in PostOrderIter(x) if x.level in self.levels])
                

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
    """
    There are two types of augmentations: Scans (how to traverse each level) and readout (how to readout the tree, i.e., DFS or BFS)
    """
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


def get_train_transforms(num_levels, downsample_factor, lowest_level, dropout, visible_levels):
    return Compose([
        OneOf([
            HorizontalRasterScan(num_levels, downsample_factor, lowest_level),
            VerticalRasterScan(num_levels, downsample_factor, lowest_level),
            HorizontalZigzagScan(num_levels, downsample_factor, lowest_level),
            VerticalZigzagScan(num_levels, downsample_factor, lowest_level)
        ]),
        TreeDropOut(num_levels, dropout),
        OneOf([
            DepthFirstReadout(levels=visible_levels),
            BreadthFirstReadout(levels=visible_levels),
            UpwardsReadout(levels=visible_levels)
        ])
    ])


def get_test_transforms(num_levels, downsample_factor, lowest_level, visible_levels):
    return Compose([
        HorizontalRasterScan(num_levels, downsample_factor, lowest_level, p_i=0, p_j=0),
        DepthFirstReadout(levels=visible_levels)
    ])


class TreeDropOut:
    def __init__(self, num_levels: int, probs: list):
        self.num_levels = num_levels
        self.probs = probs
        assert len(probs) == num_levels, 'The number of probabilities should be equal to the number of levels'

    def __call__(self, root):
        for i in reversed(range(self.num_levels)):
            nodes = [node for node in LevelOrderIter(root) if node.level == i]
            for n in nodes:
                if random.random() < self.probs[i]:
                    n.drop()
        return root


def experts_train_transforms(n_experts, num_levels, downsample_factor, lowest_level, dropout, visible_levels):
    available_transforms = [
        Compose([HorizontalRasterScan(num_levels, downsample_factor, lowest_level), TreeDropOut(num_levels, dropout), DepthFirstReadout(levels=visible_levels)]),
        Compose([VerticalRasterScan(num_levels, downsample_factor, lowest_level), TreeDropOut(num_levels, dropout), DepthFirstReadout(levels=visible_levels)]),
        Compose([HorizontalZigzagScan(num_levels, downsample_factor, lowest_level), TreeDropOut(num_levels, dropout), DepthFirstReadout(levels=visible_levels)]),
        Compose([VerticalZigzagScan(num_levels, downsample_factor, lowest_level), TreeDropOut(num_levels, dropout), DepthFirstReadout(levels=visible_levels)]),
        Compose([HorizontalRasterScan(num_levels, downsample_factor, lowest_level), TreeDropOut(num_levels, dropout), BreadthFirstReadout(levels=visible_levels)]),
        Compose([VerticalRasterScan(num_levels, downsample_factor, lowest_level), TreeDropOut(num_levels, dropout), BreadthFirstReadout(levels=visible_levels)]),
        Compose([HorizontalZigzagScan(num_levels, downsample_factor, lowest_level), TreeDropOut(num_levels, dropout), BreadthFirstReadout(levels=visible_levels)]),
        Compose([VerticalZigzagScan(num_levels, downsample_factor, lowest_level), TreeDropOut(num_levels, dropout), BreadthFirstReadout(levels=visible_levels)]),
    ]

    return available_transforms[:n_experts]


def experts_test_transforms(n_experts, num_levels, downsample_factor, lowest_level, visible_levels):
    available_transforms = [
        Compose([HorizontalRasterScan(num_levels, downsample_factor, lowest_level, p_i=0, p_j=0), DepthFirstReadout(levels=visible_levels)]),
        Compose([VerticalRasterScan(num_levels, downsample_factor, lowest_level, p_i=0, p_j=0), DepthFirstReadout(levels=visible_levels)]),
        Compose([HorizontalZigzagScan(num_levels, downsample_factor, lowest_level, p_i=0, p_j=0), DepthFirstReadout(levels=visible_levels)]),
        Compose([VerticalZigzagScan(num_levels, downsample_factor, lowest_level, p_i=0, p_j=0), DepthFirstReadout(levels=visible_levels)]),
        Compose([HorizontalRasterScan(num_levels, downsample_factor, lowest_level, p_i=0, p_j=0), BreadthFirstReadout(levels=visible_levels)]),
        Compose([VerticalRasterScan(num_levels, downsample_factor, lowest_level, p_i=0, p_j=0), BreadthFirstReadout(levels=visible_levels)]),
        Compose([HorizontalZigzagScan(num_levels, downsample_factor, lowest_level, p_i=0, p_j=0), BreadthFirstReadout(levels=visible_levels)]),
        Compose([VerticalZigzagScan(num_levels, downsample_factor, lowest_level, p_i=0, p_j=0), BreadthFirstReadout(levels=visible_levels)]),
    ]

    return available_transforms[:n_experts]


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
    
    scan = HorizontalRasterScan(num_levels=3, downsample_factor=3, lowest_level=0, p_i=0.5, p_j=0.5)
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
