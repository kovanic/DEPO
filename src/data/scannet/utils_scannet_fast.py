import sys
sys.path.append('../../../src')

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import numpy as np
from numpy.linalg import inv

import cv2
from turbojpeg import TurboJPEG
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from os import path as osp
from typing import (
    List, Tuple, Dict, Any,
    Optional, Sequence, Callable
)



class JPEGReader:
    def __init__(self):
        self.reader = TurboJPEG()
        
    def __call__(self, path):
        with open(path, 'rb') as f:
            image = self.reader.decode(f.read(), 0)
        image = cv2.resize(image, (640, 480))
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image

    
def read_depth(path):
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    depth = depth / 1000
    return depth
    

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
                 intrinsics_path: str, 
                 mode: str=None,
                 mode_rgb_read: str='rgb',
                 size: Tuple[float, float]=(640, 480),
                 calculate_flow: bool=False,
                 **kwargs):
        """Manage one scene of ScanNet Dataset.
        :param root_dir: ScanNet root directory that contains scene folders.
        :param npz_path: {scene_id}.npz path. This contains image pair information of a scene.
        :param intrinsics_path: path to intrinsics.npz
        :param mode: options are ['train', 'val', 'test'].
        :param mode_rgb_read: read in 'gray' or 'rgb' mode
        :param size: the size of depth image
        :param calculate_flow: whether to calculate optical flow
        """
        super().__init__()
        self.root_dir = root_dir

        self.mode_rgb_read = mode_rgb_read
        self.size = size
        self.calculate_flow = calculate_flow
        
        with np.load(npz_path) as data:
            self.data_names = data['name']
        self.intrinsics = np.load(intrinsics_path)
            
        self.jpeg_reader = JPEGReader()
        
    def __len__(self):
        return len(self.data_names)
    
    def _read_abs_pose(self, scene_name, name):
        path = osp.join(self.root_dir, scene_name, 'pose', f'{name}.txt')
        return read_pose(path)

    def _compute_rel_pose(self, scene_name, name0, name1):
        pose0 = self._read_abs_pose(scene_name, name0)
        pose1 = self._read_abs_pose(scene_name, name1)
        return pose1 @ inv(pose0)

    def __getitem__(self, idx):
        data_name = self.data_names[idx]
        scene_name, scene_sub_name, stem_name_0, stem_name_1 = data_name
        pair_name  = f'{scene_name}_{scene_sub_name}_{stem_name_0}_{stem_name_1}'
        scene_name = f'scene{scene_name:04d}_{scene_sub_name:02d}'

        img_name0 = osp.join(self.root_dir, scene_name, 'color', f'{stem_name_0}.jpg')
        img_name1 = osp.join(self.root_dir, scene_name, 'color', f'{stem_name_1}.jpg')
        
        image0 = self.jpeg_reader(img_name0)
        image1 = self.jpeg_reader(img_name1)

        K_0 = K_1 = self.intrinsics[scene_name].reshape(3, 3).copy()
        T_0 = self._read_abs_pose(scene_name, stem_name_0)
        T_1 = self._read_abs_pose(scene_name, stem_name_1)
        
        if self.calculate_flow:
            depth0 = read_depth(osp.join(self.root_dir, scene_name, 'depth', f'{stem_name_0}.png'))
            flow_0to1, mask = optical_flow(depth0, T_0, T_1, K_0, K_1, mask=True, normalize=False)
        
        T_0to1 = torch.tensor(self._compute_rel_pose(scene_name, stem_name_0, stem_name_1),
                              dtype=torch.float32)

        data = {
            'image0': image0,  
            'image1': image1,
            'K0': K_0.astype('float32'),
            'K1': K_1.astype('float32'),
            'T_0to1': T_0to1,
            # 'T_0': torch.from_numpy(self._read_abs_pose(scene_name, stem_name_0,)),
            # 'T_1': torch.from_numpy(self._read_abs_pose(scene_name, stem_name_1,)),
            'dataset_name': 'ScanNet',
            'scene_id': scene_name,
            'pair_id': idx,
            'pair_name': pair_name,
            'pair_names': (osp.join(scene_name, 'color', f'{stem_name_0}.jpg'),
                           osp.join(scene_name, 'color', f'{stem_name_1}.jpg'))

            }
        
        if self.calculate_flow:
            data['flow_0to1'] = torch.from_numpy(flow_0to1.astype('float32'))
            data['mask'] = torch.from_numpy(mask)
            
        return data



    