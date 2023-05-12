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
        

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
        return nn.Sequential(
                            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, bias=True),
                            nn.LeakyReLU(0.1))


class ConfidenceModule(nn.Module):
    def __init__(self):
        super(ConfidenceModule, self).__init__()
        
        self._search_size = 9
        
        self.conv_0 = conv(1, 32, kernel_size=3, stride=1, padding=0)
        self.conv_1 = conv(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv_2 = conv(32, 16, kernel_size=3, stride=1, padding=0)
        self.predict_uncertainty = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=0, bias=True)
        self.conv_3 = conv(1 + 2, 32, kernel_size=3, stride=1, padding=1)
        self.conv_4 = conv(32, 16, kernel_size=3, stride=1, padding=1)
        self.predict_uncertainty_final = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1, bias=True)
        
            
    def forward(self, x, flow, bhw):
        b, h, w = bhw
        # x is now shape b*h*w, 1, s, s 
        x = groupnorm(self.conv_0(x), b, h, w)
        x = groupnorm(self.conv_1(x), b, h, w)
        x = groupnorm(self.conv_2(x), b, h, w)
    
        uncertainty_corr = self.predict_uncertainty(x)
        uncertainty_corr = uncertainty_corr.squeeze().view(b, h, w, 1).permute(0, 3, 1, 2)    
        
        flow[:, 0, ...] =  flow[:, 0, ...] / w
        flow[:, 1, ...] =  flow[:, 1, ...] / h
        
        uncertainty_and_flow = torch.cat((uncertainty_corr, flow), 1)
        x = self.conv_4(self.conv_3(uncertainty_and_flow))
        uncertainty = self.predict_uncertainty_final(x)
        uncertainty = torch.sigmoid(uncertainty)
        return uncertainty    


