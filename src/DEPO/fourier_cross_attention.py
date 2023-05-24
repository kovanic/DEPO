import torch 
from torch import nn
from torch.fft import rfftn, irfftn, fftn, ifftn


def get_activation(activation):
    if 'leaky_relu' in activation:
        slope = float(activation.split(':')[1])
        return nn.LeakyReLU(slope)

    
class SpectralRelU(nn.Module):
    def __init__(self, H, W):
        super(SpectralReLU, self).__init__()
        W = nn.Parameter(data=torch.randn(H, W))
        B = nn.Parameter(data=torch.randn(H, W))
        
    def forward(x):
        return torch.maximum(W * x + B, torch.zeros_like(x))
        
        
def conv_block(in_dim, out_dim, H, W, activation):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 1),
        nn.LayerNorm((out_dim, H, W), elementwise_affine=True),
        get_activation(activation))
    
    
class FourierCrossAttention(nn.Module):
    """This module implements the cross-attention mechanics
    via convolutions in spectral domain. feature_q learns
    from feature_s.
    
    v1: direct frequencies real (D * 2) + imaginary (D * 2) = D * 4 -> D * 2 complex
    v2: log amplitudes and phases D * 4 -> D * 2 complex
    """
    def __init__(self, dim, H, W, activation='leaky_relu:0.1', mode='v1'):
        super(FourierCrossAttention, self).__init__()
        input_dim = dim * 4
        out_dim = dim * 2
        W_ = W // 2 + 1
        self.ca = nn.Sequential(
            conv_block(input_dim, input_dim, H, W_, activation),
            conv_block(input_dim, out_dim, H, W_, activation),
            conv_block(out_dim, out_dim, H, W_, activation))
        self.mode = mode
        self.dim = dim
        
        
    def forward(self, feature_q, feature_s, H, W):
        #feature_q, feature_s: B x HW x hid_dim
        feature_q = feature_q.transpose(1, 2).unfaltten(2, (H, W))
        feature_s = feature_s.transpose(1, 2).unfaltten(2, (H, W))
        
        ft_q = rfftn(feature_q) # B x dim x H x (⌊W / 2⌋ + 1)
        ft_s = rfftn(feature_s) # B x dim x H x (⌊W / 2⌋ + 1)
        
        if self.mode == "v1":
            ft = torch.cat([torch.real(ft_q), torch.imag(ft_q), torch.real(ft_s), torch.imag(ft_s)], dim=1)
        if self.mode == "v2":
            ft = torch.cat([torch.log(torch.abs(ft_q)), torch.angle(ft_q), torch.log(torch.abs(ft_s)), torch.angle(ft_s)], dim=1)
        transformed_ft = self.ca(ft)
        re, im = torch.split(transformed_ft, self.dim, dim=1)
        weights = irfftn(torch.complex(re, im))
        feature_q = feature_q * weights
        return feature_q.flatten(2).transpose(1, 2)