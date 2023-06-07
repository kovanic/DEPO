import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from numpy.linalg import inv
from os import path as osp
from typing import (
    List, Tuple, Dict, Any,
    Optional, Sequence, Callable
)


def read_rgb(path, mode='gray', resize=(640, 480)):
    if mode == 'rgb':
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    elif mode == 'gray':
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if resize:
        image = cv2.resize(image, resize)
    if len(image.shape) == 2:
        image = torch.from_numpy(image)[None].float()
    else:
        image = torch.from_numpy(image).permute(2, 0, 1).float()
    return image


    
def read_pose(path):
    """ Read Camera2World pose and transform it to World2Camera.
    :param pose_w2c: (4, 4)
    """
    cam2world = np.loadtxt(path)
    world2cam = inv(cam2world)
    return world2cam


class SevenScenesEvalDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 pairs_path: str,
                 scene: int=None,
                 mode_rgb_read: str='rgb',
                 **kwargs):
        super().__init__()
        
        self.root_dir = root_dir
        self.mode_rgb_read = mode_rgb_read
        pairs = pd.read_csv(pairs_path, header=None, sep=' ')
        if scene is not None:
            self.pairs = pairs[pairs[2] == scene]
        else:
            self.pairs = pairs
        self._scenes_dicitionary = {
            0: 'chess', 
            1: 'fire',
            2: 'heads',
            3: 'office',
            4: 'pumpkin',
            5: 'redkitchen',
            6: 'stairs'}        
        self._K = np.array([[585., 0., 320.], [0., 585., 240.], [0., 0., 1.]]).astype('float32')
        
    def __len__(self):
        return len(self.pairs)
    
    def _read_abs_pose(self, scene_name, name):
        path = osp.join(self.root_dir, scene_name, f'{name}.pose.txt')
        return read_pose(path)

    def _compute_rel_pose(self, scene_name, name0, name1):
        pose0 = self._read_abs_pose(scene_name, name0)
        pose1 = self._read_abs_pose(scene_name, name1)
        return pose1 @ inv(pose0)

    def __getitem__(self, idx):
        scene_info = self.pairs.iloc[idx]
        name_0 = scene_info[0].split('.')[0][1:]
        name_1 = scene_info[1].split('.')[0][1:]
        scene_name = self._scenes_dicitionary[scene_info[2]]
        
        img_name_0 = osp.join(self.root_dir, scene_name, f'{name_0}.color.png')
        img_name_1 = osp.join(self.root_dir, scene_name, f'{name_1}.color.png')
        image_0 = read_rgb(img_name_0, mode=self.mode_rgb_read, resize=False)
        image_1 = read_rgb(img_name_1, mode=self.mode_rgb_read, resize=False)

        T_0to1 = self._compute_rel_pose(scene_name, name_0, name_1)

        return {
            'image_0': image_0,  
            'image_1': image_1,
            'K_0': self._K,
            'K_1': self._K,
            'T_0to1': T_0to1.astype('float32'),
            'scene_name': scene_name,
            'pair_id': scene_name + name_0 + name_1
            }


   

