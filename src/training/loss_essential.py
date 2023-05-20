import torch
from torch.linalg import norm

import sys
sys.path.append('../src')

from utils.essential_matrix import essential_matrix_from_T
from .loss_gmflow_conf import LossGMflowWithConfidence


class LossEssential:
    def __init__(self, agg_type: str):
        self.agg_type = agg_type

    def __call__(self, T_01_gt: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        '''
        Calculate Frobenuius norm between ground-truth essential matrix and its estimated counterpart.
        :param T_01_gt: ground-truth relative pose, B x 4 x 4
        :param E: estimated essential matrix, B x 3 x 3
        '''
        E_gt = essential_matrix_from_T(T_01_gt)
        E_gt = E_gt / norm(E_gt, dim=(1, 2), keepdim=True)

        loss = torch.min(norm(E - E_gt, dim=(1, 2)), norm(E + E_gt, dim=(1, 2)))

        if self.agg_type == 'mean':
            return loss.mean()
        elif self.agg_type == 'sum':
            return loss.sum()
        else:
            return loss
        
class CombinedLoss:
    def __init__(self):
        self.essential_loss = LossEssential(agg_type='mean')
        self.nll_loss = LossGMflowWithConfidence(gamma=0.9, mode='train')
        
    def __call__(self, T_01_gt, E, preds, gt_flow, mask):
        e_loss = self.essential_loss(T_01_gt, E)
        nll_loss = self.nll_loss(preds, gt_flow, mask)
        return e_loss + nll_loss