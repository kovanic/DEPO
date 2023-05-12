import sys
sys.path.append('../../src')

import torch
from torch import nn
import torch.nn.functional as F

from matching.gmflow.gmflow.gmflow import GMFlow
from .confidence_module import ConfidenceModule
from pose_regressors.dense_baseline import DensePoseRegressor
from pose_regressors.dense_one_head import DensePoseRegressorOneHead

class GMflowDensePose(nn.Module):
    
    def __init__(self,
                 dense_module='baseline',
                 num_scales=2,
                 upsample_factor=4,
                 feature_channels=128,
                 attention_type='swin',
                 num_transformer_layers=6,
                 ffn_dim_expansion=4,
                 num_head=1,
                 fine_tuning=True,

                 #for dense predictor
                 image_size=(120, 160),
                 init_loss_weights=[0., -3.],
                 conf_module=False 
                 ):
        super(GMflowDensePose, self).__init__()
        
        self.flow_model = GMFlow(
            num_scales=num_scales,
            upsample_factor=upsample_factor,
            feature_channels=feature_channels,
            attention_type=attention_type,
            num_transformer_layers=num_transformer_layers,
            ffn_dim_expansion=ffn_dim_expansion,
            num_head=num_head,
            fine_tuning=fine_tuning)
        
        self.conf_module = conf_module
        if self.conf_module:
            self._local_window_size = 9
            self.confidence_module = ConfidenceModule()
        
        if dense_module == 'one_head':
            self.dense_pose_regressor = DensePoseRegressorOneHead(
                image_size=image_size, in_ch=4+self.conf_module, init_loss_weights=init_loss_weights
            )
        else:
            self.dense_pose_regressor = DensePoseRegressor(
                image_size=image_size, in_ch=4+self.conf_module, init_loss_weights=init_loss_weights
            )
            
        
    def forward(self, img_0, img_1, attn_splits_list, corr_radius_list, prop_radius_list):
        out_flow = self.flow_model(img_0, img_1, attn_splits_list, corr_radius_list, prop_radius_list)
        flow = out_flow.pop('flow_preds')[-1]
        B, _, H, W = flow.size()
        
        #normalize flow
        flow[:, 0, :, :]  = flow[:, 0, :, :] / W
        flow[:, 1, :, :]  = flow[:, 1, :, :] / H

        if self.conf_module:
            uncertainty = self.confidence_module(
                out_flow['local_corr'].view(B * H * W, 1, self._local_window_size, self._local_window_size),
                flow,
                bhw = (B, H, W)
            )
            flow = torch.cat((flow, uncertainty), dim=1)
        
        del out_flow['local_corr']
        
        q, t = self.dense_pose_regressor(flow)
        return q, t
    
    

    
class GMflowDensePoseDeterministic(nn.Module):
    
    def __init__(self,
                 dense_module,
                 num_scales=2,
                 upsample_factor=4,
                 feature_channels=128,
                 attention_type='swin',
                 num_transformer_layers=6,
                 ffn_dim_expansion=4,
                 num_head=1,
                 fine_tuning=True):
        
        super(GMflowDensePoseDeterministic, self).__init__()
        
        self.flow_module = GMFlow(
            num_scales=num_scales,
            upsample_factor=upsample_factor,
            feature_channels=feature_channels,
            attention_type=attention_type,
            num_transformer_layers=num_transformer_layers,
            ffn_dim_expansion=ffn_dim_expansion,
            num_head=num_head,
            fine_tuning=fine_tuning)
        
        self.dense_pose_regressor = dense_module
            
        
    def forward(self, img_0, img_1, attn_splits_list, corr_radius_list, prop_radius_list,
                K_q, K_s, scales_q, scales_s):
        out_flow = self.flow_module(img_0, img_1, attn_splits_list, corr_radius_list, prop_radius_list)
        q, t = self.dense_pose_regressor(out_flow['flow_preds'][-1], K_q, K_s, scales_q, scales_s)
        return out_flow, q, t