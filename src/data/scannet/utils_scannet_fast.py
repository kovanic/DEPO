import sys
sys.path.append('../../../src')

import torch
from torch.utils.data import Dataset

import numpy as np
from numpy.linalg import inv

import cv2
from turbojpeg import TurboJPEG

from os import path as osp
from typing import (
    List, Tuple, Dict, Any,
    Optional, Sequence, Callable
)

from utils.optical_flow_numpy import optical_flow

class JPEGReader:
    def __init__(self, size):
        self.reader = TurboJPEG()
        self.size = size
        
    def __call__(self, path):
        with open(path, 'rb') as f:
            image = self.reader.decode(f.read(), 0)
        image = cv2.resize(image, self.size)
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
                 size: Tuple[float, float]=(640, 480),
                 calculate_flow: bool=False,
                 **kwargs):
        """Manage one scene of ScanNet Dataset.
        :param root_dir: ScanNet root directory that contains scene folders.
        :param npz_path: {scene_id}.npz path. This contains image pair information of a scene.
        :param intrinsics_path: path to intrinsics.npz
        :param size: the size of depth image
        :param calculate_flow: whether to calculate optical flow
        """
        super().__init__()
        self.root_dir = root_dir
        self.calculate_flow = calculate_flow
        with np.load(npz_path) as data:
            self.data_names = data['name']
        intr = np.load(intrinsics_path)
        self.intrinsics = {f: intr[f] for f in intr.files}
        self.jpeg_reader = JPEGReader(size)

        
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

        img_name_0 = osp.join(self.root_dir, scene_name, 'color', f'{stem_name_0}.jpg')
        img_name_1 = osp.join(self.root_dir, scene_name, 'color', f'{stem_name_1}.jpg')
        
        image0 = self.jpeg_reader(img_name_0)
        image1 = self.jpeg_reader(img_name_1)

        K_0 = K_1 = self.intrinsics[scene_name].reshape(3, 3).copy()        
        if self.calculate_flow:
            depth_0 = read_depth(osp.join(self.root_dir, scene_name, 'depth', f'{stem_name_0}.png'))
            T_0 = self._read_abs_pose(scene_name, stem_name_0)
            T_1 = self._read_abs_pose(scene_name, stem_name_1)
            flow_0to1, mask = optical_flow(depth_0, T_0, T_1, K_0, K_1, mask=True, normalize=False)
            
        T_0to1 = self._compute_rel_pose(scene_name, stem_name_0, stem_name_1)

        data = {
            'image_0': image0,  
            'image_1': image1,
            'K_0': K_0.astype('float32'),
            'K_1': K_1.astype('float32'),
            'T_0to1': T_0to1.astype('float32'),
            'scene_id': scene_name,
            'pair_id': idx,
            'pair_name': pair_name,
            'pair_names': (osp.join(scene_name, 'color', f'{stem_name_0}.jpg'),
                           osp.join(scene_name, 'color', f'{stem_name_1}.jpg'))

            }
        
        if self.calculate_flow:
            data.update({
                'flow_0to1': flow_0to1,
                'mask': mask,
                # 'T_0': T_0.astype('float32'),
                # 'T_1': T_1.astype('float32'),
                # 'depth_0': depth_0.astype('float32')
            })
            
        return data


# if __name__ == '__main__':
#     import time
#     from torch.utils.data import DataLoader
#     from utils.optical_flow_batch_torch import optical_flow
    
#     train_data = ScanNetDataset(
#         root_dir='/home/project/data/ScanNet/scans/',
#         npz_path='/home/project/code/data/scannet_splits/smart_sample_train_ft.npz',
#         intrinsics_path='/home/project/ScanNet/scannet_indices/intrinsics.npz',
#         calculate_flow=True)

#     train_loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True, pin_memory=True, num_workers=2)
    

#     times = []
#     for _ in range(5):
#         start = time.time()
#         data = next(iter(train_loader))
#         for key in data.keys():
#             if key in ('image_0', 'image_1', 'K_0', 'K_1', 'depth_0', 'T_0', 'T_1', 'T_0to1', 'flow_0to1'):
#                 data[key] = data[key].cuda()
#         flow_0to1 = optical_flow(data['depth_0'], data['T_0'], data['T_1'], data['K_0'], data['K_1'])
#         end = time.time()
#         times.append(end-start)
#     print(times)
