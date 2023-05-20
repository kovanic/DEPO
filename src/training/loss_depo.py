import torch
from .loss_pose import LossPose


class LossMixedDetermininstic:
    def __init__(self, mode):
        self.mode = mode
        if mode == 'train':
            self.loss_pose = LossPose(agg_type=None)
        else:
            self.loss_pose = LossPose(agg_type=None, t_norm='l2')
        
    def __call__(self, flow, q, t, T_0to1, flow_0to1, mask, weights=None):
        if self.mode == 'train':
            t_loss, q_loss = self.loss_pose(q, t, T_0to1, weights)
            t_loss = t_loss.mean()
            q_loss = q_loss.mean()
            flow_loss = ((torch.abs(flow_0to1 - flow) * mask)).mean()
            total_loss = t_loss + q_loss + flow_loss
            return total_loss, flow_loss, q_loss, t_loss
        else:
            t_loss, q_loss = self.loss_pose(q, t, T_0to1, weights)
            flow_loss = ((torch.abs(flow_0to1 - flow) * mask)).mean((1, 2, 3))
            return flow_loss, q_loss, t_loss