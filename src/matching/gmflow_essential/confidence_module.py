# Adapted from: https://github.com/PruneTruong/DenseMatching/tree/main/models

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def groupnorm(x, B, H, W):
    _, C, K, _ = x.size()
    x = x.view(B, H * W, C, K, K)
    x = ((x - x.mean((1, 3, 4)).view(B, 1, C, 1, 1)) / x.std((1, 3, 4)).view(B, 1, C, 1, 1)).view(B*H*W, C, K, K)
    return x
        
        
def estimate_average_variance_of_mixture_density(weight_map, log_var_map):
    # Computes variance of the mixture
    proba_map = torch.nn.functional.softmax(weight_map, dim=1)

    avg_variance = torch.sum(proba_map * torch.exp(log_var_map), dim=1, keepdim=True) # shape is b,1,  h, w
    return avg_variance


def estimate_probability_of_confidence_interval_of_mixture_density(weight_map, var_map, R=1.0, gaussian=False):
    """Computes P_R of a mixture of probability distributions (with K components). See PDC-Net.
    Args:
        weight_map: weight maps of each component of the mixture. They are not softmaxed yet. (B, K, H, W)
        var_map: variance corresponding to each component, (B, K, H, W)
        R: radius for the confidence interval
        gaussian: Mixture of Gaussian or Laplace densities?
    """
    # compute P_R of the mixture
    proba_map = torch.nn.functional.softmax(weight_map, dim=1)

    if gaussian:
        p_r = torch.sum(proba_map * (1 - torch.exp(-R ** 2 / (2 * var_map))), dim=1, keepdim=True)
    else:
        # laplace distribution
        p_r = torch.sum(proba_map * (1 - torch.exp(- math.sqrt(2)*R/torch.sqrt(var_map)))**2, dim=1, keepdim=True)
    return p_r



def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
        return nn.Sequential(
                            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, bias=True),
                            nn.LeakyReLU(0.1))


class MixtureDensityEstimatorFromCorr(nn.Module):
    def __init__(self, in_channels, betas=[2, 480], output_channels=4, final_image_size=(480, 640)):
        super(MixtureDensityEstimatorFromCorr, self).__init__()
        
        self._search_size = 9
        self.betas = betas
        
        self.conv_0 = conv(in_channels, 32, kernel_size=3, stride=1, padding=0)
        self.conv_1 = conv(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv_2 = conv(32, 16, kernel_size=3, stride=1, padding=0)
        
        self.predict_uncertainty = nn.Conv2d(16, output_channels, kernel_size=3, stride=1, padding=0, bias=True)
        self.upsampler = nn.Upsample(size=final_image_size, mode='bicubic')
        
        self.conv_3 = conv(4 + 2, 32, kernel_size=3, stride=1, padding=1)
        self.conv_4 = conv(32, 16, kernel_size=3, stride=1, padding=1)
        self.predict_uncertainty_final = nn.Conv2d(16, output_channels, kernel_size=3, stride=1, padding=1, bias=True)
        
            
    def forward(self, x, flow, bhw):
        b, h, w = bhw
        # x is now shape b*h*w, 1, s, s 
        x = groupnorm(self.conv_0(x), b, h, w)
        x = groupnorm(self.conv_1(x), b, h, w)
        x = groupnorm(self.conv_2(x), b, h, w)
        
        uncertainty_corr = self.predict_uncertainty(x)
        uncertainty_corr = uncertainty_corr.squeeze().view(b, h, w, 4).permute(0, 3, 1, 2)
        uncertainty_corr = self.upsampler(uncertainty_corr)
        
        uncertainty_and_flow = torch.cat((uncertainty_corr, flow), 1)
        x = self.conv_4(self.conv_3(uncertainty_and_flow))
        uncertainty = self.predict_uncertainty_final(x)

        uncertainty[:, :2] = torch.sigmoid(uncertainty[:, :2])
        uncertainty[:, 1] = self.betas[0] + (self.betas[1] - self.betas[0]) * uncertainty[:, 1]
        return uncertainty    


