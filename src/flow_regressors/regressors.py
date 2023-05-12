import torch
from torch import nn
from torch.linalg import norm
import torch.nn.functional as F


def normalize_points(grid: torch.Tensor, K: torch.Tensor, scales: tuple=(1., 1.)):
    ''' 
    :param grid: coordinates (u, v), B x 2 x H x W
    :param K: intrinsics, B x 3 x 3
    :param scales: parameters of resizing of original image
    '''    
    fx, fy, ox, oy = K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]
    sx, sy = scales[0], scales[1]
    fx *= sx
    fy *= sy
    ox *= sx
    oy *= sy
    principal_point = torch.cat([ox[..., None], oy[..., None]], -1)[..., None, None]
    focal_length = torch.cat([fx[..., None], fy[..., None]], -1)[..., None, None]
    return (grid - principal_point) / focal_length


def create_normalized_grid(size: tuple, K: torch.Tensor, scales: tuple=(1., 1.)):
    '''Given image size, return grid for pixels positions, normalized with intrinsics (K).
    :param size: image size (B, H, W)
    :param K: intrinsics, B x 3 x 3
    :param scales: parameters of resizing of original image
    '''
    B, H, W = size
    grid = torch.meshgrid((torch.arange(H), torch.arange(W)), indexing='ij')
    grid = torch.cat((grid[1].unsqueeze(0), grid[0].unsqueeze(0)), dim=0).float().to(K.device)
    grid = grid[None, ...].repeat(B, 1, 1, 1)
    grid = normalize_points(grid, K, scales)
    return grid


def conv_block(in_ch, out_ch, kernel_size, stride, padding, norm=False, W=None, H=None, activation='leaky_relu', **kwargs):
     return nn.Sequential(
         nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, **kwargs),
         nn.LeakyReLU(0.1, inplace=True) if 'leaky_relu' else nn.ReLU(inplace=True),
         nn.LayerNorm([out_ch, H, W]) if norm else nn.Identity()
     )

def conv_block_v3(in_ch, out_ch, kernel_size, stride, padding, norm=False, W=None, H=None, activation='leaky_relu', dropout=None, **kwargs):
     return nn.Sequential(
         nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, **kwargs),
         nn.LayerNorm([out_ch, H, W]) if norm else nn.Identity(),
         nn.LeakyReLU(0.1, inplace=True) if 'leaky_relu' else nn.ReLU(inplace=True),
         nn.Dropout2d(dropout) if dropout is not None else nn.Identity()
     )

    
class DensePoseRegressorV0(nn.Module):
    def __init__(self, init_loss_weights=[0., -3.]):
        super(DensePoseRegressorV0, self).__init__()
               
        loss_weights = torch.nn.Parameter(torch.tensor(init_loss_weights))
        self.register_parameter('loss_weights', loss_weights)
        
        self.shared_decoder = nn.Sequential(
            conv_block(4, 32, 5, 1, 2),
            conv_block(32, 64, 5, 5, 0),
            conv_block(64, 64, 3, 1, 1),
            conv_block(64, 64, 3, 2, 0),
            conv_block(64, 64, 3, 1, 1),
            
            conv_block(64, 64, 1, 1, 0, groups=2),
            conv_block(64, 64, 3, 1, 1, groups=2),
            conv_block(64, 64, 3, 1, 1, groups=2),
            conv_block(64, 64, 3, 1, 0, groups=2),
            conv_block(64, 32, 3, 1, 0, groups=2),
            conv_block(32, 32, 3, 1, 0, groups=2),
            conv_block(32, 32, 3, 1, 0, groups=2),
            conv_block(32, 32, 3, 1, 0, groups=2),
            conv_block(32, 32, (1, 3), 1, 0, groups=2),
            conv_block(32, 32, (1, 3), 1, 0, groups=2)   
        )
        self.angle_conv = nn.Conv2d(32, 4, 1, 1, 0)        
        self.translation_conv = nn.Conv2d(32, 3, 1, 1, 0)

        
    def forward(self, x, K_q, K_s, scales_q, scales_s):
        # x: B x 2 x H x W, 3 channels are (u, v)
        B, _, H, W = x.size()
        x = normalize_points(x, K_s, scales_s)
        grid = create_normalized_grid((B, H, W), K_q, scales_q)
        x = torch.cat((x, grid), dim=1) #B x 4 x H x W
                
        x = self.shared_decoder(x)
        t = self.translation_conv(x).squeeze(2, 3)
        q = self.angle_conv(x) #B x 4 x 1 x 1

        q[:, 0] = torch.abs(q.clone()[:, 0])
        q = q / (q ** 2).sum(dim=1).sqrt()[:, None]
        q = q.squeeze(2, 3)
        return q, t
    
    
    
class DensePoseRegressorV1(nn.Module):
    def __init__(self, init_loss_weights=[0., -3.]):
        super(DensePoseRegressorV1, self).__init__()
               
        loss_weights = torch.nn.Parameter(torch.tensor(init_loss_weights))
        self.register_parameter('loss_weights', loss_weights)
        
        self.shared_decoder = nn.Sequential(
            conv_block(4, 64, 5, 1, 2),
            conv_block(64, 128, 5, 5, 0),
            conv_block(128, 128, 3, 1, 1),
            conv_block(128, 128, 3, 2, 0),
            conv_block(128, 128, 3, 1, 1),
            
            conv_block(128, 128, 1, 1, 0, groups=2),
            conv_block(128, 128, 3, 1, 1, groups=2),
            conv_block(128, 128, 3, 1, 1, groups=2),
            conv_block(128, 128, 3, 1, 0, groups=2),
            conv_block(128, 64, 3, 1, 0, groups=2),
            conv_block(64, 64, 3, 1, 0, groups=2),
            conv_block(64, 64, 3, 1, 0, groups=2),
            conv_block(64, 64, 3, 1, 0, groups=2),
            conv_block(64, 64, (1, 3), 1, 0, groups=2),
            conv_block(64, 64, (1, 3), 1, 0, groups=2)   
        )
        self.angle_conv = nn.Conv2d(64, 4, 1, 1, 0)        
        self.translation_conv = nn.Conv2d(64, 3, 1, 1, 0)

        
    def forward(self, x, K_q, K_s, scales_q, scales_s):
        # x: B x 2 x H x W, 3 channels are (u, v)
        B, _, H, W = x.size()
        x = normalize_points(x, K_s, scales_s)
        grid = create_normalized_grid((B, H, W), K_q, scales_q)
        x = torch.cat((x, grid), dim=1) #B x 4 x H x W
                
        x = self.shared_decoder(x)
        t = self.translation_conv(x).squeeze(2, 3)
        q = self.angle_conv(x) #B x 4 x 1 x 1

        q[:, 0] = torch.abs(q.clone()[:, 0])
        q = q / (q ** 2).sum(dim=1).sqrt()[:, None]
        q = q.squeeze(2, 3)
        return q, t

    

class DensePoseRegressorV2(nn.Module):
    def __init__(self, init_loss_weights=[0., -3.], activation='leaky_relu'):
        super(DensePoseRegressorV2, self).__init__()
               
        loss_weights = torch.nn.Parameter(torch.tensor(init_loss_weights))
        self.register_parameter('loss_weights', loss_weights)
        
        self.shared_decoder = nn.Sequential(
            conv_block(4, 64, 5, 1, 2, True, 160, 120, activation),
            conv_block(64, 128, 5, 5, 0, True, 32, 24, activation),
            conv_block(128, 128, 3, 1, 1, True, 32, 24, activation),
            conv_block(128, 128, 3, 2, 0, True, 15, 11, activation),
            conv_block(128, 128, 3, 1, 1, True, 15, 11, activation),
            
            conv_block(128, 128, 1, 1, 0, True, 15, 11, groups=2, activation=activation),
            conv_block(128, 128, 3, 1, 1, True, 15, 11, groups=2, activation=activation),
            conv_block(128, 128, 3, 1, 1, True, 15, 11, groups=2, activation=activation),
            conv_block(128, 128, 3, 1, 0, groups=2, activation=activation),
            conv_block(128, 64, 3, 1, 0, groups=2, activation=activation),
            conv_block(64, 64, 3, 1, 0, groups=2, activation=activation),
            conv_block(64, 64, 3, 1, 0, groups=2, activation=activation),
            conv_block(64, 64, 3, 1, 0, groups=2, activation=activation),
            conv_block(64, 64, (1, 3), 1, 0, groups=2, activation=activation),
            conv_block(64, 64, (1, 3), 1, 0, groups=2, activation=activation)   
        )
        self.angle_conv = nn.Conv2d(64, 4, 1, 1, 0)        
        self.translation_conv = nn.Conv2d(64, 3, 1, 1, 0)

        
    def forward(self, x, K_q, K_s, scales_q, scales_s):
        # x: B x 2 x H x W, 3 channels are (u, v)
        B, _, H, W = x.size()
        x = normalize_points(x, K_s, scales_s)
        grid = create_normalized_grid((B, H, W), K_q, scales_q)
        x = torch.cat((x, grid), dim=1) #B x 4 x H x W
                
        x = self.shared_decoder(x)
        t = self.translation_conv(x).squeeze(2, 3)
        q = self.angle_conv(x) #B x 4 x 1 x 1

        q[:, 0] = torch.abs(q.clone()[:, 0])
        q = q / (q ** 2).sum(dim=1).sqrt()[:, None]
        q = q.squeeze()
        return q, t
    
    
    
class DensePoseRegressorV3(nn.Module):
    def __init__(self, init_loss_weights=[0., -3.], activation='leaky_relu', dropout=None):
        super(DensePoseRegressorV3, self).__init__()
               
        loss_weights = torch.nn.Parameter(torch.tensor(init_loss_weights))
        self.register_parameter('loss_weights', loss_weights)
        
        self.shared_decoder = nn.Sequential(
            conv_block_v3(4, 64, 5, 1, 2, True, 160, 120, activation, dropout, bias=False),
            conv_block_v3(64, 128, 5, 5, 0, True, 32, 24, activation, dropout, bias=False),
            conv_block_v3(128, 128, 3, 1, 1, True, 32, 24, activation, dropout, bias=False),
            conv_block_v3(128, 128, 3, 2, 0, True, 15, 11, activation, dropout, bias=False),
            conv_block_v3(128, 128, 3, 1, 1, True, 15, 11, activation, dropout, bias=False),
            
            conv_block_v3(128, 128, 1, 1, 0, True, 15, 11, groups=2, activation=activation, bias=False),
            conv_block_v3(128, 128, 3, 1, 1, True, 15, 11, groups=2, activation=activation, bias=False),
            conv_block_v3(128, 128, 3, 1, 1, True, 15, 11, groups=2, activation=activation, bias=False),
            conv_block_v3(128, 128, 3, 1, 0, groups=2, activation=activation),
            conv_block_v3(128, 64, 3, 1, 0, groups=2, activation=activation),
            conv_block_v3(64, 64, 3, 1, 0, groups=2, activation=activation),
            conv_block_v3(64, 64, 3, 1, 0, groups=2, activation=activation),
            conv_block_v3(64, 64, 3, 1, 0, groups=2, activation=activation),
            conv_block_v3(64, 64, (1, 3), 1, 0, groups=2, activation=activation),
            conv_block_v3(64, 64, (1, 3), 1, 0, groups=2, activation=activation)   
        )
        self.angle_conv = nn.Conv2d(64, 4, 1, 1, 0)        
        self.translation_conv = nn.Conv2d(64, 3, 1, 1, 0)

        
    def forward(self, x, K_q, K_s, scales_q, scales_s):
        # x: B x 2 x H x W, 3 channels are (u, v)
        B, _, H, W = x.size()
        x = normalize_points(x, K_s, scales_s)
        grid = create_normalized_grid((B, H, W), K_q, scales_q)
        x = torch.cat((x, grid), dim=1) #B x 4 x H x W
                
        x = self.shared_decoder(x)
        t = self.translation_conv(x).squeeze(2, 3)
        q = self.angle_conv(x) #B x 4 x 1 x 1

        q[:, 0] = torch.abs(q.clone()[:, 0])
        q = q / (q ** 2).sum(dim=1).sqrt()[:, None]
        q = q.squeeze(2, 3)
        return q, t
    
    
    
class DensePoseRegressorV4(nn.Module):
    def __init__(self, activation='leaky_relu', dropout=None):
        super(DensePoseRegressorV4, self).__init__()
                
        self.decoder = nn.Sequential(
            conv_block_v3(4, 96, 5, 1, 2, True, 160, 120, activation, dropout, bias=False),
            conv_block_v3(96, 192, 5, 5, 0, True, 32, 24, activation, dropout, bias=False, groups=3),
            conv_block_v3(192, 192, 3, 1, 1, True, 32, 24, activation, dropout, bias=False, groups=3),
            conv_block_v3(192, 192, 3, 2, 0, True, 15, 11, activation, dropout, bias=False, groups=3),
            conv_block_v3(192, 192, 3, 1, 1, True, 15, 11, activation, dropout, bias=False, groups=3),
            
            conv_block_v3(192, 192, 1, 1, 0, True, 15, 11, activation=activation, bias=False, groups=3),
            conv_block_v3(192, 192, 3, 1, 1, True, 15, 11, activation=activation, bias=False, groups=3),
            conv_block_v3(192, 192, 3, 1, 1, True, 15, 11, activation=activation, bias=False, groups=3),
            conv_block_v3(192, 192, 3, 1, 0, activation=activation, groups=3),
            conv_block_v3(192, 96, 3, 1, 0, activation=activation, groups=3),
            conv_block_v3(96, 96, 3, 1, 0, activation=activation, groups=3),
            conv_block_v3(96, 96, 3, 1, 0, activation=activation, groups=3),
            conv_block_v3(96, 96, 3, 1, 0, activation=activation, groups=3),
            conv_block_v3(96, 96, (1, 3), 1, 0, activation=activation, groups=3),
            conv_block_v3(96, 96, (1, 3), 1, 0, activation=activation, groups=3)   
        )
        self.magnitude_conv = nn.Conv2d(32, 1, 1, 1, 0)
        self.translation_conv = nn.Conv2d(32, 3, 1, 1, 0)
        self.angle_conv = nn.Conv2d(32, 4, 1, 1, 0)        
        

    def forward(self, x, K_q, K_s, scales_q, scales_s):
        # x: B x 2 x H x W, 3 channels are (u, v)
        B, _, H, W = x.size()
        x = normalize_points(x, K_s, scales_s)
        grid = create_normalized_grid((B, H, W), K_q, scales_q)
        x = torch.cat((x, grid), dim=1) #B x 4 x H x W
                
        x = self.decoder(x)
        t = self.translation_conv(x[:, :32, ...]).squeeze(2, 3)
        t = t / norm(t, ord=2, dim=1, keepdim=True)
        
        m = self.magnitude_conv(x[:, 32:64, ...]).squeeze(2, 3)
        
        q = self.angle_conv(x[:, 64:, ...]).squeeze(2, 3) #B x 4 x 1 x 1 -> B x 4
        q[:, 0] = torch.abs(q.clone()[:, 0])
        q = q / norm(q, ord=2, dim=1, keepdim=True)
       
        return q, t, m

    
    
class DensePoseRegressorV5(nn.Module):
    def __init__(self, activation='leaky_relu', dropout=None):
        super(DensePoseRegressorV5, self).__init__()
                
        self.decoder = nn.Sequential(
            conv_block_v3(4, 64, 5, 1, 2, True, 160, 120, activation, dropout, bias=False),
            conv_block_v3(64, 128, 5, 5, 0, True, 32, 24, activation, dropout, bias=False, groups=2),
            conv_block_v3(128, 128, 3, 1, 1, True, 32, 24, activation, dropout, bias=False, groups=2),
            conv_block_v3(128, 128, 3, 2, 0, True, 15, 11, activation, dropout, bias=False, groups=2),
            conv_block_v3(128, 128, 3, 1, 1, True, 15, 11, activation, dropout, bias=False, groups=2),
            
            conv_block_v3(128, 128, 1, 1, 0, True, 15, 11, activation=activation, bias=False, groups=2),
            conv_block_v3(128, 128, 3, 1, 1, True, 15, 11, activation=activation, bias=False, groups=2),
            conv_block_v3(128, 128, 3, 1, 1, True, 15, 11, activation=activation, bias=False, groups=2),
            conv_block_v3(128, 128, 3, 1, 0, activation=activation, groups=2),
            conv_block_v3(128, 64, 3, 1, 0, activation=activation, groups=2),
            conv_block_v3(64, 64, 3, 1, 0, activation=activation, groups=2),
            conv_block_v3(64, 64, 3, 1, 0, activation=activation, groups=2),
            conv_block_v3(64, 64, 3, 1, 0, activation=activation, groups=2),
            conv_block_v3(64, 64, (1, 3), 1, 0, activation=activation, groups=2),
            conv_block_v3(64, 64, (1, 3), 1, 0, activation=activation, groups=2)   
        )
        
        self.translation_conv = nn.Conv2d(32, 3, 1, 1, 0)
        self.angle_conv = nn.Conv2d(32, 4, 1, 1, 0)        
        
        
    def forward(self, x, K_q, K_s, scales_q, scales_s):
        # x: B x 2 x H x W, 3 channels are (u, v)
        B, _, H, W = x.size()
        x = normalize_points(x, K_s, scales_s)
        grid = create_normalized_grid((B, H, W), K_q, scales_q)
        x = torch.cat((x, grid), dim=1) #B x 4 x H x W
                
        x = self.decoder(x)
        t = self.translation_conv(x[:, :32, ...]).squeeze(2, 3)
        q = self.angle_conv(x[:, 32:, ...]).squeeze(2, 3) #B x 4 x 1 x 1 -> B x 4
        q[:, 0] = torch.abs(q.clone()[:, 0])
        q = q / norm(q, ord=2, dim=1, keepdim=True)
        return q, t
    
    
    
class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, H, W, kernel_size, stride=1, **kwargs):
        super(ResNetBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False, **kwargs),
            nn.LayerNorm([out_ch, H, W]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=stride, bias=False, **kwargs),
            nn.LayerNorm([out_ch, H // stride + H % stride, W // stride + W % stride])
        )
        
        self.conv1x1 = nn.Conv2d(in_ch, out_ch, 1, 1, 0)
        self.pooling = nn.AdaptiveAvgPool2d((H // stride + H % stride, W // stride + W % stride))
        self.final_act = nn.LeakyReLU(0.1, inplace=True)
        
    def forward(self, x):
        x = self.final_act(self.pooling(self.conv1x1(x)) + self.block(x))
        return x


    
class ResNetBlockSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, H, W, kernel_size, stride=1, **kwargs):
        super(ResNetBlockSeparable, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, groups=in_ch),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=False, **kwargs),
            nn.LayerNorm([out_ch, H, W]),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=stride, groups=in_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, padding=0, bias=False, **kwargs),
            nn.LayerNorm([out_ch, H // stride + H % stride, W // stride + W % stride])
        )
        
        self.conv1x1 = nn.Conv2d(in_ch, out_ch, 1, 1, 0)
        self.pooling = nn.AdaptiveAvgPool2d((H // stride + H % stride, W // stride + W % stride))
        self.final_act = nn.LeakyReLU(0.1, inplace=True)
        
    def forward(self, x):
        x = self.final_act(self.pooling(self.conv1x1(x)) + self.block(x))
        return x

    
    
class DensePoseRegressorV6(nn.Module):
    def __init__(self):
        super(DensePoseRegressorV6, self).__init__()
        
        self.decoder = nn.Sequential(
            conv_block_v3(4, 64, 5, 1, 2, True, 160, 120, bias=False),
            ResNetBlock(64, 64, 120, 160, 3, 1, groups=2),
            ResNetBlock(64, 64, 120, 160, 3, 2, groups=2),
            ResNetBlock(64, 64, 60, 80, 3, 2, groups=2),
            ResNetBlock(64, 64, 30, 40, 3, 2, groups=2),
            ResNetBlock(64, 64, 15, 20, 3, 2, groups=2),
            ResNetBlock(64, 64, 8, 10, 3, 2, groups=2),    
            nn.Conv2d(64, 64, (4, 5), 1, 0),
        )
        self.translation_conv = nn.Conv2d(32, 3, 1, 1, 0)
        self.angle_conv = nn.Conv2d(32, 4, 1, 1, 0)
        
        
    def forward(self, x, K_q, K_s, scales_q, scales_s):
        # x: B x 2 x H x W, 3 channels are (u, v)
        B, _, H, W = x.size()
        x = normalize_points(x, K_s, scales_s)
        grid = create_normalized_grid((B, H, W), K_q, scales_q)
        x = torch.cat((x, grid), dim=1) #B x 4 x H x W
                
        x = self.decoder(x)
        t = self.translation_conv(x[:, :32, ...]).squeeze(2, 3)
        q = self.angle_conv(x[:, 32:, ...]).squeeze(2, 3) #B x 4 x 1 x 1 -> B x 4
        q[:, 0] = torch.abs(q.clone()[:, 0])
        q = q / norm(q, ord=2, dim=1, keepdim=True)
        return q, t
    
    
    
class DensePoseRegressorV7(nn.Module):
    def __init__(self, init_loss_weights=[0., -3.]):
        super(DensePoseRegressorV7, self).__init__()
        
        loss_weights = torch.nn.Parameter(torch.tensor(init_loss_weights))
        self.register_parameter('loss_weights', loss_weights)
        
        self.decoder = nn.Sequential(
            conv_block_v3(4, 64, 5, 1, 2, True, 160, 120, bias=False),
            ResNetBlock(64, 64, 120, 160, 3, 1),
            ResNetBlock(64, 64, 120, 160, 3, 2),
            ResNetBlock(64, 64, 60, 80, 3, 2, groups=2),
            ResNetBlock(64, 64, 30, 40, 3, 2, groups=2),
            ResNetBlock(64, 64, 15, 20, 3, 2, groups=2),
            ResNetBlock(64, 64, 8, 10, 3, 2, groups=2),    
            nn.Conv2d(64, 64, (4, 5), 1, 0),
        )
        self.translation_conv = nn.Conv2d(32, 3, 1, 1, 0)
        self.angle_conv = nn.Conv2d(32, 4, 1, 1, 0)
        
        
    def forward(self, x, K_q, K_s, scales_q, scales_s):
        # x: B x 2 x H x W, 3 channels are (u, v)
        B, _, H, W = x.size()
        x = normalize_points(x, K_s, scales_s)
        grid = create_normalized_grid((B, H, W), K_q, scales_q)
        x = torch.cat((x, grid), dim=1) #B x 4 x H x W
                
        x = self.decoder(x)
        t = self.translation_conv(x[:, :32, ...]).squeeze(2, 3)
        q = self.angle_conv(x[:, 32:, ...]).squeeze(2, 3) #B x 4 x 1 x 1 -> B x 4
        q[:, 0] = torch.abs(q.clone()[:, 0])
        q = q / norm(q, ord=2, dim=1, keepdim=True)
        return q, t


    
class DensePoseRegressorV8(nn.Module):
    def __init__(self):
        super(DensePoseRegressorV8, self).__init__()
        
        self.decoder = nn.Sequential(
            conv_block_v3(4, 64, 5, 1, 2, True, 160, 120, bias=False),
            ResNetBlockSeparable(64, 64, 120, 160, 3, 1, groups=2),
            ResNetBlockSeparable(64, 64, 120, 160, 3, 2, groups=2),
            ResNetBlockSeparable(64, 64, 60, 80, 3, 2, groups=2),
            ResNetBlockSeparable(64, 64, 30, 40, 3, 2, groups=2),
            ResNetBlockSeparable(64, 64, 15, 20, 3, 2, groups=2),
            ResNetBlockSeparable(64, 64, 8, 10, 3, 2, groups=2),    
            nn.Conv2d(64, 64, (4, 5), 1, 0),
        )
        self.translation_conv = nn.Conv2d(32, 3, 1, 1, 0)
        self.angle_conv = nn.Conv2d(32, 4, 1, 1, 0)
        
        
    def forward(self, x, K_q, K_s, scales_q, scales_s):
        # x: B x 2 x H x W, 3 channels are (u, v)
        B, _, H, W = x.size()
        x = normalize_points(x, K_s, scales_s)
        grid = create_normalized_grid((B, H, W), K_q, scales_q)
        x = torch.cat((x, grid), dim=1) #B x 4 x H x W
                
        x = self.decoder(x)
        t = self.translation_conv(x[:, :32, ...]).squeeze(2, 3)
        q = self.angle_conv(x[:, 32:, ...]).squeeze(2, 3) #B x 4 x 1 x 1 -> B x 4
        q[:, 0] = torch.abs(q.clone()[:, 0])
        q = q / norm(q, ord=2, dim=1, keepdim=True)
        return q, t
    

    
class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module
    with given atrous_rates and out_channels for each head
    """
    
    def __init__(self, in_ch, out_ch, num_ch=16, kernel_size=3, atrous_rates=[1, 2, 3, 5]):
        super(ASPP, self).__init__()
        
        self.conv = nn.ModuleList()
        
        self.conv.extend(nn.ModuleList(
                            nn.Sequential(
                                nn.Conv2d(in_ch, num_ch, kernel_size, dilation=rate, padding="same", bias=False),
                                nn.LayerNorm([num_ch, 120, 160]),
                                nn.LeakyReLU(0.1),
                            ) for rate in atrous_rates))
        
        self.conv.append(nn.Conv2d(in_ch, num_ch, kernel_size=1))
        self.conv.append(nn.Sequential(
                            nn.AdaptiveAvgPool2d(1),
                            nn.Conv2d(in_ch, num_ch, kernel_size=1)
        ))
        
        self.compression = nn.Sequential(
                            nn.Conv2d(num_ch * (len(atrous_rates) + 2), out_ch, kernel_size=1, bias=False),
                            nn.LayerNorm([out_ch, 120, 160]),
                            nn.LeakyReLU(0.1)
        )  


    def forward(self, x):       
        conv_output = [block(x) for block in self.conv[:-1]]
        pool_output = self.conv[-1](x)
        pool_output = F.interpolate(pool_output, size=x.shape[-2:], mode="bilinear", align_corners=False)
        conv_output.append(pool_output)
        conv_output = torch.cat(conv_output, dim=1)
        
        res = self.compression(conv_output)
        return res          
    

    
class DensePoseRegressorV9(nn.Module):
    def __init__(self, init_loss_weights=[0., -3.]):
        super(DensePoseRegressorV9, self).__init__()
        
        loss_weights = torch.nn.Parameter(torch.tensor(init_loss_weights))
        self.register_parameter('loss_weights', loss_weights)
        
        self.decoder = nn.Sequential(
            ASPP(4, 64), 
            ResNetBlock(64, 64, 120, 160, 3, 1),
            ResNetBlock(64, 64, 120, 160, 3, 2),
            ResNetBlock(64, 64, 60, 80, 3, 2, groups=2),
            ResNetBlock(64, 64, 30, 40, 3, 2, groups=2),
            ResNetBlock(64, 64, 15, 20, 3, 2, groups=2),
            ResNetBlock(64, 64, 8, 10, 3, 2, groups=2),    
            nn.Conv2d(64, 64, (4, 5), 1, 0),
        )
        self.translation_conv = nn.Conv2d(32, 3, 1, 1, 0)
        self.angle_conv = nn.Conv2d(32, 4, 1, 1, 0)
        
        
    def forward(self, x, K_q, K_s, scales_q, scales_s):
        # x: B x 2 x H x W, 3 channels are (u, v)
        B, _, H, W = x.size()
        x = normalize_points(x, K_s, scales_s)
        grid = create_normalized_grid((B, H, W), K_q, scales_q)
        x = torch.cat((x, grid), dim=1) #B x 4 x H x W
                
        x = self.decoder(x)
        t = self.translation_conv(x[:, :32, ...]).squeeze(2, 3)
        q = self.angle_conv(x[:, 32:, ...]).squeeze(2, 3) #B x 4 x 1 x 1 -> B x 4
        q[:, 0] = torch.abs(q.clone()[:, 0])
        q = q / norm(q, ord=2, dim=1, keepdim=True)
        return q, t