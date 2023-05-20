"""Partially adapted from LoFTR code: https://github.com/zju3dv/LoFTR/tree/master/src/utils"""
"""Partially adapted from LoFTR code: https://github.com/zju3dv/LoFTR/tree/master/src/datasets/scannet.py"""

import sys
sys.path.append('/home/project/code/src')

import cv2

import torch
from torch.utils.data import Dataset

import numpy as np
from numpy.linalg import inv
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from os import path as osp
from typing import (
    List, Tuple, Dict, Any,
    Optional, Sequence, Callable)
from tqdm.auto import trange, tqdm


from utils.optical_flow_numpy import optical_flow
from pathlib import Path
from multiprocessing import cpu_count, Pool,  RLock
from skimage.measure import block_reduce


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


def getPixelsForInterp(img): 
    """
    Source: https://stackoverflow.com/questions/59748831/torch-interpolate-missing-values
    Calculates a mask of pixels neighboring invalid values - 
       to use for interpolation. 
    """
    # mask invalid pixels
    invalid_mask = np.isnan(img) + (img == 0) 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    #dilate to mark borders around invalid regions
    dilated_mask = cv2.dilate(invalid_mask.astype('uint8'), kernel, 
                      borderType=cv2.BORDER_CONSTANT, borderValue=int(0))

    # pixelwise "and" with valid pixel mask (~invalid_mask)
    masked_for_interp = dilated_mask *  ~invalid_mask
    return masked_for_interp.astype('bool'), invalid_mask


def fillMissingValues(target_for_interp, copy=True,
                      interpolator=LinearNDInterpolator): 
    '''Adapted from: https://stackoverflow.com/questions/59748831/torch-interpolate-missing-values
    Fill missing values of depth bilinearly for convex skull of image and fill all remaining ones
    with NN interpolation.
    '''
    if copy: 
        target_for_interp = target_for_interp.copy()
        
    # Mask pixels for interpolation
    mask_for_interp, invalid_mask = getPixelsForInterp(target_for_interp)

    # Interpolate only holes, only using these pixels
    points = np.argwhere(mask_for_interp)
    values = target_for_interp[mask_for_interp]
    interp = interpolator(points, values)
    target_for_interp[invalid_mask] = interp(np.argwhere(invalid_mask))
    
    # Interpolate points out of convex skull
    interp = NearestNDInterpolator(points, values)
    out_of_skull = np.isnan(target_for_interp)
    missing = np.argwhere(out_of_skull )
    target_for_interp[out_of_skull] = interp(missing)
    return target_for_interp


class ScanNetFlow(Dataset):
    def __init__(self,
                 root_dir: str,
                 npz_path: str,
                 intrinsics_path: str, 
                 task: List=[None, None],
                 mode: str='train',
                 min_overlap_score: float=0.4,
                 reduce: bool=False,
                 **kwargs):
        """Manage one scene of ScanNet Dataset.
        :param root_dir: ScanNet root directory that contains scene folders.
        :param npz_path: {scene_id}.npz path. This contains image pair information of a scene.
        :param intrinsics_path: path to intrinsics.npz
        :param mode: options are ['train', 'val', 'test'].
        :param mode_rgb_read: read in 'gray' or 'rgb' mode
        :param min_overlap_score: minimum covisibility to include a pair of images in train
        :param reduce: whether to reduce flow
        """
        super().__init__()
        self.root_dir = root_dir
        self.reduce = reduce
        with np.load(npz_path) as data:
            self.data_names = data['name'][task[0]:task[1]]
            if 'score' in data.keys() and mode not in ['val' or 'test']:
                kept_mask = data['score'][task[0]:task[1]] > min_overlap_score
                self.data_names = self.data_names[kept_mask]
        self.intrinsics = np.load(intrinsics_path)

    def __len__(self):
        return len(self.data_names)
    
    def _read_abs_pose(self, scene_name, name):
        path = osp.join(self.root_dir, scene_name, 'pose', f'{name}.txt')
        return read_pose(path)
    
    def _compute_rel_pose(self, T_0, T_1):
        return T_1 @ inv(T_0)
    
    def __getitem__(self, idx):
        data_name = self.data_names[idx]
        scene_name, scene_sub_name, stem_name_0, stem_name_1 = data_name
        scene_name = f'scene{scene_name:04d}_{scene_sub_name:02d}'

        K_0 = K_1 = self.intrinsics[scene_name].reshape(3, 3)
        T_0 = self._read_abs_pose(scene_name, stem_name_0)
        T_1 = self._read_abs_pose(scene_name, stem_name_1)
        
        depth_0 = read_depth(osp.join(self.root_dir, scene_name, 'depth', f'{stem_name_0}.png'))
        try:
            depth_0 = fillMissingValues(depth_0, copy=False)
        except:
            print(f'{scene_name}_{stem_name_0}_{stem_name_1}')
        flow_0to1, _ = optical_flow(depth_0, T_0, T_1, K_0, K_1, mask=False, normalize=False)
        
        T_0to1 = self._compute_rel_pose(T_0, T_1)
        
        if self.reduce:
            flow_0to1 = block_reduce(flow_0to1, (1, 4, 4), np.mean) / 4
          
        return {
                'flow': flow_0to1.astype('float32'),
                'rel_pose': T_0to1, 
                'K_s': K_1.astype('float32'),
                'K_q': K_0.astype('float32'),
                'pair_id': f'{scene_name}_{stem_name_0}_{stem_name_1}'
               }

    
if __name__ == '__main__':
    
    jobs = cpu_count()
    N = np.load('/home/project/ScanNet/flow_train_indices_subset.npz')['name'].shape[0]
    s = np.linspace(0, N, jobs+1, dtype=int)
    tasks = [[s[i], s[i+1]] for i in range(len(s)-1)]
   
    def work(task):
        folder_to_save = Path('/home/project/data_sample/flow/train')
        folder_to_save.mkdir(parents=True, exist_ok=True)
        
        data = ScanNetFlow(
            task=task,
            root_dir='/home/project/data/ScanNet/scans/',
            npz_path='/home/project/ScanNet/flow_train_indices_subset.npz',
            intrinsics_path='/home/project/ScanNet/scannet_indices/intrinsics.npz',
            mode='train'
        )
        for i in trange(len(data), disable=False, lock_args=None, position=None): 
            out = data[i]
            pair_id = out.pop('pair_id')
            np.savez(folder_to_save / (pair_id + '.npz'), **out)
            
    pool = Pool(jobs)    
    pool.map(work, tasks)
    pool.close()
    pool.join()
    
    
# if __name__ == '__main__':
    
#     jobs = cpu_count()
#     N = np.load('/home/project/ScanNet/flow_val_indices_subset.npz')['name'].shape[0]
#     s = np.linspace(0, N, jobs+1, dtype=int)
#     tasks = [[s[i], s[i+1]] for i in range(len(s)-1)]
   
#     def work(task):
#         folder_to_save = Path('/home/project/data_sample/flow/val')
#         folder_to_save.mkdir(parents=True, exist_ok=True)
        
#         data = ScanNetFlow(
#             task=task,
#             root_dir='/home/project/data/ScanNet/scans/',
#             npz_path='/home/project/ScanNet/flow_val_indices_subset.npz',
#             intrinsics_path='/home/project/ScanNet/scannet_indices/intrinsics.npz',
#             mode='val'
#         )
#         for i in trange(len(data), disable=False, lock_args=None, position=None): 
#             out = data[i]
#             pair_id = out.pop('pair_id')
#             np.savez(folder_to_save / (pair_id + '.npz'), **out)
            
#     pool = Pool(jobs)    
#     pool.map(work, tasks)
#     pool.close()
#     pool.join()