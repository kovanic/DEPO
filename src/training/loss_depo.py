import torch
from torch import nn
from .loss_pose import LossPose, LossPoseRelative
from torch.linalg import norm


class LossMixedDetermininstic:
    def __init__(self, mode, add_l2=False):
        self.mode = mode
        self.add_l2 = add_l2
        if mode == 'train':
            self.loss_pose = LossPose(agg_type=None, add_l2=add_l2)
        else:
            self.loss_pose = LossPose(agg_type=None, t_norm='l2')
        
    def __call__(self, flow, q, t, T_0to1, flow_0to1, mask, weights=None):
        if self.mode == 'train':
            t_loss, q_loss, t_angle_loss = self.loss_pose(q, t, T_0to1, weights)
            t_loss = t_loss.mean()
            q_loss = q_loss.mean()
            flow_loss = ((torch.abs(flow_0to1 - flow) * mask)).mean()
            total_loss = t_loss + q_loss + flow_loss 
            if self.add_l2:
                t_angle_loss = t_angle_loss.mean()
                total_loss += t_angle_loss  
            return total_loss, flow_loss, q_loss, t_loss
        else:
            t_loss, q_loss, _ = self.loss_pose(q, t, T_0to1, weights)
            # flow_loss = ((torch.abs(flow_0to1 - flow) * mask)).mean((1, 2, 3))
            return None, q_loss, t_loss
        


class LossMixedDetermininsticWeighted:
    def __init__(self, mode, weights, add_l2=False):
        self.mode = mode
        self.add_l2 = add_l2
        if weights is not None:
            self.weights = nn.Parameter(torch.tensor(weights))
        
        if mode == 'train':
            self.loss_pose = LossPose(agg_type=None, add_l2=add_l2)
        else:
            self.loss_pose = LossPose(agg_type=None, t_norm='l2')
        
    def __call__(self, flow, q, t, T_0to1, flow_0to1, mask):
        if self.mode == 'train':
            t_loss, q_loss, t_angle_loss = self.loss_pose(q, t, T_0to1, weights=None)

            # weights = torch.minimum(self.weights, torch.zeros_like(self.weights))
            t_loss = t_loss.mean() * torch.exp(-self.weights[0]) + self.weights[0]
            q_loss = q_loss.mean() * torch.exp(-self.weights[1]) + self.weights[1] 
            flow_loss = ((torch.abs(flow_0to1 - flow) * mask)).mean() * torch.exp(-self.weights[2]) + self.weights[2]
            total_loss = t_loss + q_loss + flow_loss 
            if self.add_l2:
                t_angle_loss = t_angle_loss.mean() * torch.exp(-self.weights[3]) + self.weights[3]
                total_loss += t_angle_loss  
            return total_loss, flow_loss, q_loss, t_loss
        else:
            t_loss, q_loss, _ = self.loss_pose(q, t, T_0to1, weights=None)
            return None, q_loss, t_loss
        
        
        
        
class LossMixedRelativeWeighted:
    def __init__(self, mode, weights=None):
        self.mode = mode
        if weights is not None:
            self.weights = nn.Parameter(torch.tensor(weights))
        self.loss_pose = LossPoseRelative(agg_type=None)
        
    def __call__(self, flow, q, t, T_0to1, flow_0to1, mask):
        if self.mode == 'train':
            t_loss, q_loss = self.loss_pose(q, t, T_0to1)
            t_loss = t_loss.mean() * torch.exp(-self.weights[0]) + self.weights[0]
            q_loss = q_loss.mean() * torch.exp(-self.weights[1]) + self.weights[1] 
            flow_loss = ((torch.abs(flow_0to1 - flow) * mask)).mean() * torch.exp(-self.weights[2]) + self.weights[2]
            total_loss = t_loss + q_loss + flow_loss 
            return total_loss, flow_loss, q_loss, t_loss
        else:
            t_loss, q_loss = self.loss_pose(q, t, T_0to1)
            return None, q_loss, t_loss