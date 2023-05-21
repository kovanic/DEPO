 import torch
from torch import nn
import torch.nn.functional as F

from .nll_losses import NLLMixtureLaplace


# class LossGMflowWithConfidence():
    
#     def __init__(self, gamma=0.9):
#         self.loss_mixture = NLLMixtureLaplace(reduction='mean')
#         self.gamma = gamma
        
#     def __call__(self, preds, gt_flow, mask):
#         loss = (
#             self.gamma ** 3 * (torch.abs(preds['flow_preds'][0] - gt_flow) * mask).mean() + 
#             self.gamma ** 2 * (torch.abs(preds['flow_preds'][1] - gt_flow) * mask).mean() + 
#             self.gamma * self.loss_mixture(gt_flow, preds['flow_preds'][2], torch.log(preds['var'][:, :2]), preds['var'][:, 2:], mask=mask) + 
#                     self.loss_mixture(gt_flow, preds['flow_preds'][3], torch.log(preds['var'][:, :2]), preds['var'][:, 2:], mask=mask)
#         )
#         return loss
    
    
class LossGMflowWithConfidence:
    
    def __init__(self, gamma=0.9, mode='train'):
        self.loss_mixture = NLLMixtureLaplace(reduction='mean')
        self.gamma = gamma
        self.mode = mode
        
    def __call__(self, preds, gt_flow, mask):
        
        if self.mode == 'train':
            l1 = self.gamma * self.loss_mixture(gt_flow, preds['flow_preds'][2], torch.log(preds['var'][:, :2]), preds['var'][:, 2:], mask=mask)
            l2 = self.loss_mixture(gt_flow, preds['flow_preds'][3], torch.log(preds['var'][:, :2]), preds['var'][:, 2:], mask=mask)
            return l1+l2
        else:
            return self.loss_mixture(gt_flow, preds['flow_preds'][0], torch.log(preds['var'][:, :2]), preds['var'][:, 2:], mask=mask)
