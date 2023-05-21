import torch
from torch import nn
from torch.linalg import norm
import torch.nn.functional as F


def conv_block(in_ch, out_ch, kernel_size, stride, padding, norm=False, H=None, W=None, activation='leaky_relu', dropout=None, **kwargs):
     return nn.Sequential(
         nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, **kwargs),
         nn.LayerNorm([out_ch, H, W]) if norm else nn.Identity(),
         nn.LeakyReLU(0.1, inplace=True) if 'leaky_relu' else nn.ReLU(inplace=True),
         nn.Dropout2d(dropout) if dropout is not None else nn.Identity()
     )
    
    
class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, H, W, kernel_size, stride=1, **kwargs):
        super(ResNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False, **kwargs),
            nn.LayerNorm([out_ch, H, W]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=stride, **kwargs)
        )
        self.residual =  nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, 1, 0),
            nn.AdaptiveAvgPool2d((H // stride + H % stride, W // stride + W % stride)))
        self.final_norm = nn.LayerNorm([out_ch, H // stride + H % stride, W // stride + W % stride])
        self.final_act = nn.LeakyReLU(0.1, inplace=True)
        
    def forward(self, x):
        return self.final_act(self.final_norm(self.residual(x) + self.block(x)))


    
class DensePoseRegressorV1(nn.Module):
    def __init__(self, in_ch=4):
        super(DensePoseRegressorV1, self).__init__()
        
        self.decoder = nn.Sequential(
            conv_block(in_ch, 64, 5, 1, 2, True, 60, 80, bias=False),
            ResNetBlock(64, 64, 60, 80, 3, 2),
            ResNetBlock(64, 64, 30, 40, 3, 2, groups=2),
            ResNetBlock(64, 64, 15, 20, 3, 2, groups=2),
            ResNetBlock(64, 64, 8, 10, 3, 2, groups=2),    
            nn.Conv2d(64, 64, (4, 5), 1, 0, groups=2),
            nn.LeakyReLU(0.1)
        )
        self.translation_conv = nn.Conv2d(32, 3, 1, 1, 0)
        self.angle_conv = nn.Conv2d(32, 4, 1, 1, 0)
        
        
    def forward(self, x):               
        x = self.decoder(x)
        t = self.translation_conv(x[:, :32, ...]).squeeze(2).squeeze(2)
        q = self.angle_conv(x[:, 32:, ...]).squeeze(2).squeeze(2) #B x 4 x 1 x 1 -> B x 4
        q[:, 0] = torch.abs(q.clone()[:, 0])
        q = q / norm(q, ord=2, dim=1, keepdim=True)
        return q, t
    
    
    
class DensePoseRegressorV2(nn.Module):
    def __init__(self, in_ch=4, init_loss_weights=[0., -3.]):
        super(DensePoseRegressorV2, self).__init__()
        
        loss_weights = torch.nn.Parameter(torch.tensor(init_loss_weights))
        self.register_parameter('loss_weights', loss_weights)
        
        self.decoder = nn.Sequential(
            conv_block(in_ch, 64, 5, 1, 2, True, 60, 80, bias=False),
            ResNetBlock(64, 64, 60, 80, 3, 2, groups=2),
            ResNetBlock(64, 64, 30, 40, 3, 2, groups=2),
            ResNetBlock(64, 64, 15, 20, 3, 2, groups=2),
            ResNetBlock(64, 64, 8, 10, 3, 2, groups=2),    
            nn.Conv2d(64, 64, (4, 5), 1, 0, groups=2),
            nn.LeakyReLU(0.1)
        )
        self.translation_conv = nn.Conv2d(32, 3, 1, 1, 0)
        self.angle_conv = nn.Conv2d(32, 4, 1, 1, 0)
        
        
    def forward(self, x):
        x = self.decoder(x)
        t = self.translation_conv(x[:, :32, ...]).squeeze(2).squeeze(2)
        q = self.angle_conv(x[:, 32:, ...]).squeeze(2).squeeze(2) #B x 4 x 1 x 1 -> B x 4
        q[:, 0] = torch.abs(q.clone()[:, 0])
        q = q / norm(q, ord=2, dim=1, keepdim=True)
        return q, t


    
class DensePoseRegressorV3(nn.Module):
    def __init__(self, in_ch):
        super(DensePoseRegressorV3, self).__init__()
    
        self.decoder = nn.Sequential(
            ResNetBlock(in_ch, 64, 60, 80, 3, 2, groups=2),
            ResNetBlock(64, 64, 30, 40, 3, 2, groups=2),
            ResNetBlock(64, 64, 15, 20, 3, 2, groups=2),
            ResNetBlock(64, 64, 8, 10, 3, 2, groups=2),    
            nn.Conv2d(64, 64, (4, 5), 1, 0),
            nn.LeakyReLU(0.1)
        )
        self.translation_conv = nn.Conv2d(32, 3, 1, 1, 0)
        self.angle_conv = nn.Conv2d(32, 4, 1, 1, 0)
        
        
    def forward(self, x):               
        x = self.decoder(x)
        t = self.translation_conv(x[:, :32, ...]).squeeze(2).squeeze(2)
        q = self.angle_conv(x[:, 32:, ...]).squeeze(2).squeeze(2) #B x 4 x 1 x 1 -> B x 4
        q[:, 0] = torch.abs(q.clone()[:, 0])
        q = q / norm(q, ord=2, dim=1, keepdim=True)
        return q, t
    
    
    
class DensePoseRegressorV4(nn.Module):
    def __init__(self, in_ch, init_loss_weights=[0., -3.]):
        super(DensePoseRegressorV4, self).__init__()
        
        loss_weights = torch.nn.Parameter(torch.tensor(init_loss_weights))
        self.register_parameter('loss_weights', loss_weights)
        
        self.decoder = nn.Sequential(
            ResNetBlock(in_ch, 64, 60, 80, 3, 2, groups=2),
            ResNetBlock(64, 64, 30, 40, 3, 2, groups=2),
            ResNetBlock(64, 64, 15, 20, 3, 2, groups=2),
            ResNetBlock(64, 64, 8, 10, 3, 2, groups=2),    
            nn.Conv2d(64, 64, (4, 5), 1, 0,  groups=2),
            nn.LeakyReLU(0.1)
        )
        self.translation_conv = nn.Conv2d(32, 3, 1, 1, 0)
        self.angle_conv = nn.Conv2d(32, 4, 1, 1, 0)
        
        
    def forward(self, x):
        x = self.decoder(x)
        t = self.translation_conv(x[:, :32, ...]).squeeze(2).squeeze(2)
        q = self.angle_conv(x[:, 32:, ...]).squeeze(2).squeeze(2) #B x 4 x 1 x 1 -> B x 4
        q[:, 0] = torch.abs(q.clone()[:, 0])
        q = q / norm(q, ord=2, dim=1, keepdim=True)
        return q, t
    
    
    
class DensePoseRegressorV5(nn.Module):
    def __init__(self, in_ch):
        super(DensePoseRegressorV5, self).__init__()
        
        self.in_ch = in_ch
        
        self.decoder = nn.Sequential(
            ResNetBlock(in_ch, in_ch, 60, 80, 3, 1),
            ResNetBlock(in_ch, in_ch, 60, 80, 3, 2, groups=2),
            ResNetBlock(in_ch, in_ch, 30, 40, 3, 2, groups=2),
            ResNetBlock(in_ch, in_ch, 15, 20, 3, 2, groups=2),
            ResNetBlock(in_ch, in_ch, 8, 10, 3, 2, groups=2),    
            nn.Conv2d(in_ch, in_ch, (4, 5), 1, 0, groups=2),
            nn.LeakyReLU(0.1)
        )
        self.translation_conv = nn.Conv2d(in_ch // 2, 3, 1, 1, 0)
        self.angle_conv = nn.Conv2d(in_ch // 2, 4, 1, 1, 0)
        
        
    def forward(self, x):               
        x = self.decoder(x)
        t = self.translation_conv(x[:, :self.in_ch // 2, ...]).squeeze(2).squeeze(2)
        q = self.angle_conv(x[:, self.in_ch // 2:, ...]).squeeze(2).squeeze(2) #B x 4 x 1 x 1 -> B x 4
        q[:, 0] = torch.abs(q.clone()[:, 0])
        q = q / norm(q, ord=2, dim=1, keepdim=True)
        return q, t
    
    
    
class DensePoseRegressorV6(nn.Module):
    def __init__(self, in_ch):
        super(DensePoseRegressorV6, self).__init__()
        
        self.in_ch = in_ch
        
        self.decoder = nn.Sequential(
            ResNetBlock(in_ch, in_ch, 60, 80, 3, 1),
            ResNetBlock(in_ch, in_ch, 60, 80, 3, 2),
            ResNetBlock(in_ch, in_ch, 30, 40, 3, 2),
            ResNetBlock(in_ch, in_ch, 15, 20, 3, 2),
            ResNetBlock(in_ch, in_ch, 8, 10, 3, 2, groups=2),    
            nn.Conv2d(in_ch, in_ch, (4, 5), 1, 0, groups=2),
            nn.LeakyReLU(0.1)
        )
        self.translation_conv = nn.Conv2d(in_ch // 2, 3, 1, 1, 0)
        self.angle_conv = nn.Conv2d(in_ch // 2, 4, 1, 1, 0)
        
        
    def forward(self, x):               
        x = self.decoder(x)
        t = self.translation_conv(x[:, :self.in_ch // 2, ...]).squeeze(2).squeeze(2)
        q = self.angle_conv(x[:, self.in_ch // 2:, ...]).squeeze(2).squeeze(2) #B x 4 x 1 x 1 -> B x 4
        q[:, 0] = torch.abs(q.clone()[:, 0])
        q = q / norm(q, ord=2, dim=1, keepdim=True)
        return q, t