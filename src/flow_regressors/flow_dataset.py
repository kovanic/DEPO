import torch
from torch.utils.data import Dataset
import numpy as np
from numpy.linalg import inv
from pathlib import Path
import os 
from skimage.measure import block_reduce


class FlowDataset(Dataset):
    def __init__(self, root_path, intrinsics_path):
        super().__init__()
        self.pathes = [os.path.join(root_path, file) for file in os.listdir(root_path)]
        self.intrinsics = np.load(intrinsics_path)
        
    def __len__(self):
        return len(self.pathes)
    
    def __getitem__(self, idx):
        path = self.pathes[idx]
        data = np.load(path)
        scene_name = '_'.join(path.split('/')[-1].split('.')[0].split('_')[:2])
        K = self.intrinsics[scene_name].reshape(3, 3).astype('float32')
        flow = block_reduce(data['flow'], (1, 4, 4), np.mean) / 4
        return {
            'flow': flow,
            'rel_pose': data['rel_pose'],
            'K_q': K,
            'K_s': K
        }

def read_pose(path: np.ndarray):
    """ Read ScanNet's Camera2World pose and transform it to World2Camera.
    :param pose_w2c: (4, 4)
    """
    cam2world = np.loadtxt(path, delimiter=' ')
    world2cam = inv(cam2world)
    return world2cam


    
class FlowDatasetV1(Dataset):
    def __init__(self, root_dir, root_pose_dir, intrinsics_path):
        super().__init__()
        self.pathes = [os.path.join(root_dir, file) for file in os.listdir(root_dir)]
        self.root_pose_dir = root_pose_dir
        self.intrinsics = np.load(intrinsics_path)
        
    def __len__(self):
        return len(self.pathes)
    
    def _read_abs_pose(self, scene_name, name):
        path = os.path.join(self.root_pose_dir, scene_name, 'pose', f'{name}.txt')
        return read_pose(path)

    def _compute_rel_pose(self, scene_name, name0, name1):
        pose0 = self._read_abs_pose(scene_name, name0)
        pose1 = self._read_abs_pose(scene_name, name1)
        return pose1 @ inv(pose0)
    
    
    def __getitem__(self, idx):
        path = self.pathes[idx]
        data = torch.load(path)
        
        scene, subscene, name_0, name_1 = os.path.basename(path).split('.')[0].split('_')
        scene_name = f'scene{int(scene):04d}_{int(subscene):02d}'

        T_0to1 = torch.tensor(self._compute_rel_pose(scene_name, name_0, name_1),
                              dtype=torch.float32)
        
        K = self.intrinsics[scene_name].reshape(3, 3).astype('float32')
    
        return {
            'flow': data,
            'rel_pose': T_0to1,
            'K_q': K,
            'K_s': K
        }