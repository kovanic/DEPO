import sys
sys.path.append('../../src')

import torch
from torch import nn
import torch.nn.functional as F

from matching.gmflow.gmflow.gmflow import GMFlow
from .confidence_module import MixtureDensityEstimatorFromCorr, estimate_probability_of_confidence_interval_of_mixture_density
from pose_regressors.weighted8points import EssRegressor


class GMFlowEssential(nn.Module):
    def __init__(self,
                 num_scales=2,
                 upsample_factor=4,
                 feature_channels=128,
                 attention_type='swin',
                 num_transformer_layers=6,
                 ffn_dim_expansion=4,
                 num_head=1,
                 fine_tuning=False,
                 # confidence estimator
                 in_channels=1,
                 betas=[2, 480],
                 output_channels=4,
                 final_image_size=(480, 640),
                 # essential matrix regressor
                 n_samples=10000):
        super(GMFlowEssential, self).__init__()
        
        self.flow_model = GMFlow(
            num_scales=num_scales,
            upsample_factor=upsample_factor,
            feature_channels=feature_channels,
            attention_type=attention_type,
            num_transformer_layers=num_transformer_layers,
            ffn_dim_expansion=ffn_dim_expansion,
            num_head=num_head,
            fine_tuning=fine_tuning)
        
        self._local_window_size = 9
        self.confidence_module = MixtureDensityEstimatorFromCorr(
            in_channels=in_channels,
            betas=betas,
            output_channels=output_channels,
            final_image_size=final_image_size)
        self.final_image_size = final_image_size
        
        self.essential_regeressor = EssRegressor(n_samples=n_samples)


    def forward(self, img_0, img_1, K_0, K_1, attn_splits_list, corr_radius_list, prop_radius_list):
        out_flow = self.flow_model(img_0, img_1, attn_splits_list, corr_radius_list, prop_radius_list)
        B, *_ = out_flow['local_corr'].size()
        
        var = self.confidence_module(
            out_flow['local_corr'].view(B * (self.final_image_size[0] // 4) * (self.final_image_size[1] // 4),
                                        1, self._local_window_size, self._local_window_size),
            out_flow['flow_preds'][-1],
            bhw = (B, (self.final_image_size[0] // 4), (self.final_image_size[1] // 4))
        )
        
        del out_flow['local_corr']
        out_flow['var'] = var
        uncertainty = estimate_probability_of_confidence_interval_of_mixture_density(
            var[:, 2:, ...], var[:, :2, ...], R=4.0, gaussian=False
        )
        del var

        E = self.essential_regeressor(torch.cat((out_flow['flow_preds'][-1], uncertainty), dim=1), K_0, K_1) 
        return E, out_flow
        