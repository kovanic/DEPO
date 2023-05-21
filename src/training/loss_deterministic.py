from .loss_pose import LossPose
import torch

class DeterministicLossMixed:
    def __init__(self, gamma=0.9):
        self.gamma = gamma
        self.loss_pose = LossPose(agg_type='mean')
    
    def __call__(self, flow, mask, T_0to1, flow_preds, q, t, weights):
        l1 = self.loss_pose(q, t, T_0to1, weights)
        l2 = 0
        for i in range(4):
            l2 += ((torch.abs(flow_preds[i] - flow) * mask)).mean() * (self.gamma ** (3 - i)) * 0.25
        return l1 + l2