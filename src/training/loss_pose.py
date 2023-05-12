import torch
import numpy as np
from scipy.spatial.transform import Rotation
from torch.linalg import norm
# from scipy.stats import ortho_group - for debugging

class LossPose:
    '''Calculate weighted loss between ground-truth and predicted translation
    and set of rotations (lying in quaternion space).
    :param agg_type: type of used aggregation: None, mean, sum'''
    
    def __init__(self, agg_type=None):
        self.agg_type = agg_type
        
    def __call__(self, q: torch.Tensor, t: torch.Tensor, T_0to1: torch.Tensor, weights: torch.Tensor=None):
        '''
        :param q: predicted normalized quaternions tensor, order [w, x, y, z], B x 4
        :param t: predicted translations tensor, B x 3
        :param T_0to1: ground-truth realtive pose, B x 4 x 4
        :param weights: learnable weighting factors, 2
        :return: loss value
        '''
        R_gt, t_gt = T_0to1[:, :3, :3], T_0to1[:, :3, 3]

        # Note: scipy returns normalized quaternion in [x, y, z, w] order
        # Note: det(R) = 1, otherwise it is not from SO(3)
        q_gt = torch.from_numpy(
            Rotation.from_matrix(R_gt).as_quat()
        ).to(q.device)
        
        q_gt = q_gt[:, [3, 0, 1, 2]]
        q_gt[q_gt[:, 0] < 0] = -q_gt[q_gt[:, 0] < 0] 
       
        t_gt = t_gt.to(t.device)

        l1 = torch.abs(t_gt - t).sum(dim=1)
        l2 = torch.abs(q_gt - q).sum(dim=1) 
        l3 = 0.
        
        if weights is not None:
            l1 = l1 * torch.exp(-weights[0]) + 3*weights[0]
            l2 = l2 * torch.exp(-weights[1]) + 4*weights[1]
            
            if len(weights) == 3:
                l3 = ((t_gt / norm(t_gt, ord=2, dim=1, keepdim=True) - t / norm(t, ord=2, dim=1, keepdim=True)) ** 2).sum(dim=1)
                l3 = l3 * torch.exp(-weights[2]) + 3*weights[2]
                
        if self.agg_type == 'mean':
            return (l1 + l2 + l3).mean()
        elif self.agg_type == 'sum':
            return (l1 + l2 + l3).sum()
        else:
            return l1, l2
        
    
class LossPoseV1:
    '''Calculate loss between ground-truth and predicted normalized translation, magnitude and
    rotations (lying in quaternion space).
    :param agg_type: type of used aggregation: None, mean, sum
    :param mode: train/test
    '''
    
    def __init__(self, agg_type=None, mode='train'):
        self.agg_type = agg_type
        self.mode = mode
        
    def __call__(self, q: torch.Tensor, t_n: torch.Tensor, t_m: torch.Tensor, T_0to1: torch.Tensor):
        '''
        :param q: predicted normalized quaternions tensor, order [w, x, y, z], B x 4
        :param t_n: predicted normalized translations tensor, B x 3
        :param t_m: tensor of predicted magnitudes, B x 1
        :param T_0to1: ground-truth realtive pose, B x 4 x 4
        :param weights: learnable weighting factors, 2
        :return: loss value
        '''
        R_gt, t_gt = T_0to1[:, :3, :3], T_0to1[:, :3, 3]

        # Note: scipy returns normalized quaternion in [x, y, z, w] order
        # Note: det(R) = 1, otherwise it is not from SO(3)
        q_gt = torch.from_numpy(
            Rotation.from_matrix(R_gt).as_quat()
        ).to(q.device)
        
        q_gt = q_gt[:, [3, 0, 1, 2]]
        q_gt[q_gt[:, 0] < 0] = -q_gt[q_gt[:, 0] < 0] 
       
        t_gt = t_gt.to(t_n.device)
        
        if self.mode in {'test', 'val'}:
            l1 = torch.abs(t_gt - t_m * t_n).sum(dim=1)
            l2 = torch.abs(q_gt - q).sum(dim=1) 
            return l1, l2
        
        l1 = torch.abs(norm(t_gt, ord=2, dim=1, keepdim=True) - t_m).sum(dim=1)
        l2 = torch.abs(q_gt - q).sum(dim=1) 
        l3 = ((t_gt / norm(t_gt, ord=2, dim=1, keepdim=True) - t_n) ** 2).sum(dim=1)
               
        if self.agg_type == 'mean':
            return (l1 + l2 + l3).mean()
        elif self.agg_type == 'sum':
            return (l1 + l2 + l3).sum()
        else:
            return l1, l2, l3
    