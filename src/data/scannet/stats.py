import sys
sys.path.append('../../../src')

import numpy as np
from numpy.linalg import inv

import torch
from torch.utils.data import Dataset

from os import path as osp
from typing import (
    List, Tuple, Dict, Any,
    Optional, Sequence, Callable
)


def read_pose(path: np.ndarray):
    """ Read ScanNet's Camera2World pose and transform it to World2Camera.
    :param pose_w2c: (4, 4)
    """
    cam2world = np.loadtxt(path, delimiter=' ')
    world2cam = inv(cam2world)
    return world2cam


class ScanNetDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 npz_path: str,
                 **kwargs):
        """Manage one scene of ScanNet Dataset.
        :param root_dir: ScanNet root directory that contains scene folders.
        :param npz_path: {scene_id}.npz path. This contains image pair information of a scene.
        """
        super().__init__()
        self.root_dir = root_dir
        with np.load(npz_path) as data:
            self.data_names = data['name']
 
    def __len__(self):
        return len(self.data_names)
    
    def _read_abs_pose(self, scene_name, name):
        path = osp.join(self.root_dir, scene_name, 'pose', f'{name}.txt')
        return read_pose(path)

    def _compute_rel_pose(self, scene_name, name_0, name_1):
        pose0 = self._read_abs_pose(scene_name, name_0)
        pose1 = self._read_abs_pose(scene_name, name_1)
        return pose1 @ inv(pose0)

    def __getitem__(self, idx):
        data_name = self.data_names[idx]
        scene_name, scene_sub_name, stem_name_0, stem_name_1 = data_name
        pair_name  = f'{scene_name}_{scene_sub_name}_{stem_name_0}_{stem_name_1}'
        scene_name = f'scene{scene_name:04d}_{scene_sub_name:02d}'

        T_0to1 = self._compute_rel_pose(scene_name, stem_name_0, stem_name_1)

        return T_0to1