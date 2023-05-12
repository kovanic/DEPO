import sys
sys.path.append('../../src')

import torch
from torch import nn
import torch.nn.functional as F

from matching.gmflow.gmflow.gmflow import GMFlow
from .mod_uncertainty import (MixtureDensityEstimatorFromCorr,
                              estimate_probability_of_confidence_interval_of_mixture_density)


class GMflowWithConfidence(nn.Module):
    
    def __init__(self,
                 device='cpu',
                 num_scales=2,
                 upsample_factor=4,
                 feature_channels=128,
                 attention_type='swin',
                 num_transformer_layers=6,
                 ffn_dim_expansion=4,
                 num_head=1,
                 local_window_size=9,
                 #for confidence module
                 conf_in_channels=1,
                 conf_betas=[2, 480],
                 conf_output_channels=4,
                 conf_upsample_factor=4,
                 upsample=False,
                 final_image_size=(480, 640),
                 ):
        super(GMflowWithConfidence, self).__init__()
        
        self.local_window_size = 9
        self.final_image_size = final_image_size
        self.flow_model = GMFlow(
            num_scales=num_scales,
            upsample_factor=upsample_factor,
            feature_channels=feature_channels,
            attention_type=attention_type,
            num_transformer_layers=num_transformer_layers,
            ffn_dim_expansion=ffn_dim_expansion,
            num_head=num_head)
        
        self.confidence_module = MixtureDensityEstimatorFromCorr(
            in_channels=conf_in_channels,
            betas=conf_betas,
            output_channels=conf_output_channels,
            final_image_size=final_image_size
        )
                
        
    def forward(self, img_0, img_1, attn_splits_list, corr_radius_list, prop_radius_list):
        out_flow = self.flow_model(img_0, img_1, attn_splits_list, corr_radius_list, prop_radius_list)
        B, *_ = out_flow['local_corr'].size()
        
        var = self.confidence_module(
            out_flow['local_corr'].view(B * (self.final_image_size[0] // 4) * (self.final_image_size[1] // 4),
                                        1, self.local_window_size, self.local_window_size),
            out_flow['flow_preds'][-1],
            bhw = (B, self.final_image_size[0] // 4, self.final_image_size[1] // 4)
        )
        
        del out_flow['local_corr']
        out_flow['var'] = var
        return out_flow