import torch
from torch import nn
import torch.nn.functional as F

import os.path as osp
import sys
file_name = osp.abspath(__file__)
dir_name = osp.dirname(file_name)
sys.path.append(dir_name)

from .quadtree_attention import QuadtreeAttention
from .twins_backbone import (
    pcpvt_small_v0_partial, pcpvt_base_v0_partial, pcpvt_large_v0_partial,
    alt_gvt_small_partial, alt_gvt_base_partial, alt_gvt_large_partial
)

from .pose_regressors import DensePoseRegressorV1, DensePoseRegressorV2, DensePoseRegressorV3, DensePoseRegressorV4, DensePoseRegressorV5, DensePoseRegressorV6



def normalize_imgs(imgs):
    # loaded images are in [0, 255]
    # normalize by ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(imgs.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(imgs.device)
    imgs = (imgs / 255. - mean) / std
    return imgs



def normalize_points(grid: torch.Tensor, K: torch.Tensor, scales: torch.Tensor):
    ''' 
    :param grid: coordinates (u, v), B x 2 x H x W
    :param K: intrinsics, B x 3 x 3
    :param scales: parameters of resizing of original image, B x 2
    '''    
    fx, fy, ox, oy = K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]
    sx, sy = scales[: ,0], scales[:, 1]
    fx = fx * sx
    fy = fy * sy
    ox = ox * sx
    oy = ox * sy
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
    grid = torch.meshgrid((torch.arange(H), torch.arange(W)))
    grid = torch.cat((grid[1].unsqueeze(0), grid[0].unsqueeze(0)), dim=0).float().to(K.device)
    grid = grid[None, ...].repeat(B, 1, 1, 1)
    grid = normalize_points(grid, K, scales)
    return grid



class DEPO_v1(nn.Module):
    """Self-attention -> Cross-attention -> (R, t) & flow.
    :param mode: {'flow&pose', 'flow->pose', 'pose'}
    """
    def __init__(self, self_encoder, cross_encoder, pose_regressor, mode, hid_dim, num_emb, upsample_factor=1):
        super(DEPO_v1, self).__init__()
         
        self.hid_dim = hid_dim
        self.num_emb = num_emb
        self.mode = mode
        self.upsample_factor = upsample_factor
        
        self.self_encoder = self_encoder
        self.cross_encoder = cross_encoder        
        self.geometry_decoder = nn.Sequential(
            nn.Linear(hid_dim, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, hid_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hid_dim, num_emb),
            nn.LeakyReLU(0.1),
        )
        
        if self.mode in ('flow&pose', 'pose'):
            self.intrinsics_mlp = nn.Sequential(
                nn.Linear(20, num_emb),
                nn.LeakyReLU(0.1),
                nn.Linear(num_emb, num_emb * 2),
                nn.LeakyReLU(0.1),
                nn.Linear(num_emb * 2, num_emb * 2))
        
        if 'flow' in self.mode:
            self.flow_regressor = nn.Sequential(
                nn.Conv2d(num_emb, num_emb, 3, 1, 1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(num_emb, num_emb // 2, 1, 1, 0),
                nn.LeakyReLU(0.1),
                nn.Conv2d(num_emb // 2, num_emb // 2, 3, 1, 1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(num_emb // 2, 2, 1, 1, 0),
            )
            
            # convex upsampling (Source: GMFlow):
            self.upsampler = nn.Sequential(nn.Conv2d(2 + hid_dim, 256, 3, 1, 1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0))
            
        self.pose_regressor = pose_regressor
      
    
    def upsample_flow(self, flow, feature):
        # (Source: GMFlow)
        concat = torch.cat((flow, feature), dim=1)
        mask = self.upsampler(concat)
        
        b, flow_channel, h, w = flow.shape
        mask = mask.view(b, 1, 9, self.upsample_factor, self.upsample_factor, h, w)  # [B, 1, 9, K, K, H, W]
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(self.upsample_factor * flow, [3, 3], padding=1) #cut windows in flow [18, H, W]
        up_flow = up_flow.view(b, flow_channel, 9, 1, 1, h, w)  # [B, 2, 9, 1, 1, H, W]

        up_flow = torch.sum(mask * up_flow, dim=2)  # [B, 2, K, K, H, W]
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [B, 2, K, H, K, W]
        up_flow = up_flow.reshape(b, flow_channel, self.upsample_factor * h,
                                  self.upsample_factor * w)  # [B, 2, K*H, K*W]
        return up_flow

                      
    def calibration_to_vector_(self, K_q, K_s, scales_q, scales_s):
        '''Organize calibration parameters in vector:
        [f_x^q, f_y^q, o_x^q, o_y^q, f_x^s, f_y^s, o_x^s, o_y^s, s_x^q, s_y^q, s_x^s, s_y^s,
         f_x^q * s_x^q, f_y^q * s_y^q, o_x^q * s_x^q, o_y^q * s_y^q,
         f_x^s * s_x^s, f_y^s * s_y^s, o_x^s * s_x^s, o_y^s * s_y^s]
        :param K_{q, s}: calibration matrix, B x 3 x 3
        :scales_{q, s}: the multiplier of initial image size, expected to be in [0, 1], B x 2
        '''
        return torch.cat((
            K_q[:, [0, 1, 0, 1], [0, 1, 2, 2]],
            K_s[:, [0, 1, 0, 1], [0, 1, 2, 2]],
            K_q[:, [0, 0], [0, 2]] * scales_q[:, 0, None],
            K_q[:, [1, 1], [1, 2]] * scales_q[:, 1, None],
            K_s[:, [0, 0], [0, 2]] * scales_s[:, 0, None],
            K_s[:, [1, 1], [1, 2]] * scales_s[:, 1, None],
            scales_q, scales_s
        ), dim=1) / 1000.
    
            
    def forward(self, img_q, img_s, K_q, K_s, scales_q, scales_s, H=60, W=80):    
        #Apply self-attention module
        B = img_q.size(0)
        imgs = normalize_imgs(torch.cat((img_q, img_s), dim=0))
        features = self.self_encoder(imgs)
        
        #Apply cross-attention module
        features_q, features_s = features.split(B) # N x hid_dim x H x W 
        features_q = features_q.contiguous().view((B, self.hid_dim, H*W)).transpose(2, 1)
        features_s = features_s.contiguous().view((B, self.hid_dim, H*W)).transpose(2, 1)
        features_qc = self.cross_encoder(features_q, features_s, H, W) # B x HW x hid_dim
        
        #Hidden geometry extraction
        hidden_geometry = self.geometry_decoder(features_q - features_qc) # B x HW x num_emb
        hidden_geometry = hidden_geometry.transpose(2, 1).contiguous().view(B, -1, H, W)
        
    
        if self.mode in ('pose', 'flow&pose'):
            calibration_vector = self.calibration_to_vector_(K_q, K_s, scales_q, scales_s)
            calibration_parameters = self.intrinsics_mlp(calibration_vector) # num_emb*2
            mu, sigma = calibration_parameters[:, :self.num_emb, None, None], torch.exp(calibration_parameters[:, self.num_emb:, None, None])
            pose_regressor_input = (hidden_geometry - mu) / sigma
            q, t = self.pose_regressor(pose_regressor_input, K_q, K_s, scales_q, scales_s)
        
        if 'flow' in self.mode:
            flow_coarse = self.flow_regressor(hidden_geometry)
            flow = self.upsample_flow(flow_coarse, (features_q - features_qc).transpose(2, 1).contiguous().view(B, -1, H, W))
            
        if self.mode == 'pose':
            return None, q, t
            
        if self.mode == 'flow&pose':
            return flow, q, t
            
        if self.mode == 'flow->pose':
            flow_coarse = normalize_points(flow_coarse, K_s, scales_s)
            grid = create_normalized_grid((B, H, W), K_q, scales_q)
            flow_coarse = torch.cat((flow_coarse, grid), dim=1) #B x 4 x H x W
            q, t = self.pose_regressor(flow_coarse, K_q, K_s, scales_q, scales_s)
            return flow, q, t
            
            
############################Configurations############################
######################################################################
            
def depo_v0():
    self_encoder = pcpvt_large_v0_partial(img_size=(640, 480))
    self_encoder.load_state_dict(torch.load(osp.join(dir_name, 'weights_external/pcpvt_large.pth')), strict=False)
    cross_encoder = QuadtreeAttention(dim=128, num_heads=8, topks=[16, 16, 8], scale=3)
    pose_regressor = DensePoseRegressorV3(128)
    return DEPO(
        self_encoder=self_encoder,
        cross_encoder=cross_encoder,
        pose_regressor=pose_regressor,
        hid_dim=128,
        num_emb=128,
        mode='flow&pose',
        upsample_factor=8)


def depo_v1():
    self_encoder = pcpvt_large_v0_partial(img_size=(640, 480))
    self_encoder.load_state_dict(torch.load(osp.join(dir_name, 'weights_external/pcpvt_large.pth')), strict=False)
    cross_encoder = QuadtreeAttention(dim=128, num_heads=8, topks=[16, 16, 8], scale=3)
    pose_regressor = DensePoseRegressorV5(128)
    return DEPO(
        self_encoder=self_encoder,
        cross_encoder=cross_encoder,
        pose_regressor=pose_regressor,
        hid_dim=128,
        num_emb=128,
        mode='pose')
            

def depo_v2():
    self_encoder = pcpvt_large_v0_partial(img_size=(640, 480))
    self_encoder.load_state_dict(torch.load(osp.join(dir_name, 'weights_external/pcpvt_large.pth')), strict=False)
    cross_encoder = QuadtreeAttention(dim=128, num_heads=8, topks=[16, 16, 8], scale=3)
    pose_regressor = DensePoseRegressorV1(4)
    return DEPO_v1(
        self_encoder=self_encoder,
        cross_encoder=cross_encoder,
        pose_regressor=pose_regressor,
        hid_dim=128,
        num_emb=128,
        upsample_factor=8,
        mode='flow->pose')


def depo_v3():
    self_encoder = pcpvt_large_v0_partial(img_size=(640, 480))
    self_encoder.load_state_dict(torch.load(osp.join(dir_name, 'weights_external/pcpvt_large.pth')), strict=False)
    cross_encoder = QuadtreeAttention(dim=128, num_heads=8, topks=[16, 16, 8], scale=3)
    pose_regressor = DensePoseRegressorV5(128)
    return DEPO(
        self_encoder=self_encoder,
        cross_encoder=cross_encoder,
        pose_regressor=pose_regressor,
        hid_dim=128,
        num_emb=128,
        mode='flow&pose',
        upsample_factor=8)

    
def depo_v4():
    self_encoder = alt_gvt_large_partial(img_size=(640, 480))
    self_encoder.load_state_dict(torch.load(osp.join(dir_name, 'weights_external/pvt_large.pth')), strict=False)
    cross_encoder = QuadtreeAttention(dim=256, num_heads=8, topks=[16, 16, 8], scale=3)
    pose_regressor = DensePoseRegressorV5(128)
    return DEPO_v1(
        self_encoder=self_encoder,
        cross_encoder=cross_encoder,
        pose_regressor=pose_regressor,
        hid_dim=256,
        num_emb=128,
        mode='',
        upsample_factor=8)

def depo_v6():
    self_encoder = pcpvt_large_v0_partial(img_size=(640, 480))
    self_encoder.load_state_dict(torch.load(osp.join(dir_name, 'weights_external/pcpvt_large.pth')), strict=False)
    cross_encoder = QuadtreeAttention(dim=128, num_heads=8, topks=[16, 16, 8], scale=3)
    pose_regressor = DensePoseRegressorV6(128)
    return DEPO_v1(
        self_encoder=self_encoder,
        cross_encoder=cross_encoder,
        pose_regressor=pose_regressor,
        hid_dim=128,
        num_emb=128,
        mode='pose')



############################Legacy####################################
######################################################################


#MISTAKE with geometry_embedding (this is actually another 1x1 conv), corrected in DEPO_v1
class DEPO(nn.Module):
    """Self-attention -> Cross-attention -> (R, t) & flow.
    :param mode: {'flow&pose', 'flow->pose', 'pose'}
    """
    def __init__(self, self_encoder, cross_encoder, pose_regressor, mode, hid_dim, num_emb, upsample_factor=1):
        super(DEPO, self).__init__()
        
        geometry_embedding = nn.Parameter(torch.randn((hid_dim, num_emb)))
        self.register_parameter('geometry_embedding', geometry_embedding)
        
        self.hid_dim = hid_dim
        self.num_emb = num_emb
        self.mode = mode
        self.upsample_factor = upsample_factor
        
        self.self_encoder = self_encoder
        self.cross_encoder = cross_encoder        
        self.geometry_decoder = nn.Sequential(
            nn.Linear(hid_dim, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, hid_dim))
        
        if self.mode in ('flow&pose', 'pose'):
            self.intrinsics_mlp = nn.Sequential(
                nn.Linear(20, num_emb),
                nn.LeakyReLU(0.1),
                nn.Linear(num_emb, num_emb * 2),
                nn.LeakyReLU(0.1),
                nn.Linear(num_emb * 2, num_emb * 2))
        
        if 'flow' in self.mode:
            self.flow_regressor = nn.Sequential(
                nn.Conv2d(num_emb, num_emb, 3, 1, 1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(num_emb, num_emb // 2, 1, 1, 0),
                nn.LeakyReLU(0.1),
                nn.Conv2d(num_emb // 2, num_emb // 2, 3, 1, 1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(num_emb // 2, 2, 1, 1, 0),
            )
            
            # convex upsampling (Source: GMFlow):
            self.upsampler = nn.Sequential(nn.Conv2d(2 + hid_dim, 256, 3, 1, 1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0))
            
        self.pose_regressor = pose_regressor
      
    
    def upsample_flow(self, flow, feature):
        # (Source: GMFlow)
        concat = torch.cat((flow, feature), dim=1)
        mask = self.upsampler(concat)
        
        b, flow_channel, h, w = flow.shape
        mask = mask.view(b, 1, 9, self.upsample_factor, self.upsample_factor, h, w)  # [B, 1, 9, K, K, H, W]
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(self.upsample_factor * flow, [3, 3], padding=1) #cut windows in flow [18, H, W]
        up_flow = up_flow.view(b, flow_channel, 9, 1, 1, h, w)  # [B, 2, 9, 1, 1, H, W]

        up_flow = torch.sum(mask * up_flow, dim=2)  # [B, 2, K, K, H, W]
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [B, 2, K, H, K, W]
        up_flow = up_flow.reshape(b, flow_channel, self.upsample_factor * h,
                                  self.upsample_factor * w)  # [B, 2, K*H, K*W]
        return up_flow

                      
    def calibration_to_vector_(self, K_q, K_s, scales_q, scales_s):
        '''Organize calibration parameters in vector:
        [f_x^q, f_y^q, o_x^q, o_y^q, f_x^s, f_y^s, o_x^s, o_y^s, s_x^q, s_y^q, s_x^s, s_y^s,
         f_x^q * s_x^q, f_y^q * s_y^q, o_x^q * s_x^q, o_y^q * s_y^q,
         f_x^s * s_x^s, f_y^s * s_y^s, o_x^s * s_x^s, o_y^s * s_y^s]
        :param K_{q, s}: calibration matrix, B x 3 x 3
        :scales_{q, s}: the multiplier of initial image size, expected to be in [0, 1], B x 2
        '''
        return torch.cat((
            K_q[:, [0, 1, 0, 1], [0, 1, 2, 2]],
            K_s[:, [0, 1, 0, 1], [0, 1, 2, 2]],
            K_q[:, [0, 0], [0, 2]] * scales_q[:, 0, None],
            K_q[:, [1, 1], [1, 2]] * scales_q[:, 1, None],
            K_s[:, [0, 0], [0, 2]] * scales_s[:, 0, None],
            K_s[:, [1, 1], [1, 2]] * scales_s[:, 1, None],
            scales_q, scales_s
        ), dim=1) / 1000.
    
            
    def forward(self, img_q, img_s, K_q, K_s, scales_q, scales_s, H=60, W=80):    
        #Apply self-attention module
        B = img_q.size(0)
        imgs = normalize_imgs(torch.cat((img_q, img_s), dim=0))
        features = self.self_encoder(imgs)
        
        #Apply cross-attention module
        features_q, features_s = features.split(B) # N x hid_dim x H x W 
        features_q = features_q.contiguous().view((B, self.hid_dim, H*W)).transpose(2, 1)
        features_s = features_s.contiguous().view((B, self.hid_dim, H*W)).transpose(2, 1)
        features_qc = self.cross_encoder(features_q, features_s, H, W) # B x HW x hid_dim
        
        #Hidden geometry extraction
        hidden_geometry = self.geometry_decoder(features_q - features_qc) @ self.geometry_embedding # B x HW x num_emb
        hidden_geometry = hidden_geometry.transpose(2, 1).contiguous().view(B, -1, H, W)
        
    
        if self.mode in ('pose', 'flow&pose'):
            calibration_vector = self.calibration_to_vector_(K_q, K_s, scales_q, scales_s)
            calibration_parameters = self.intrinsics_mlp(calibration_vector) # num_emb*2
            mu, sigma = calibration_parameters[:, :self.num_emb, None, None], torch.exp(calibration_parameters[:, self.num_emb:, None, None])
            pose_regressor_input = (hidden_geometry - mu) / sigma
            q, t = self.pose_regressor(pose_regressor_input, K_q, K_s, scales_q, scales_s)
        
        if 'flow' in self.mode:
            flow_coarse = self.flow_regressor(hidden_geometry)
            flow = self.upsample_flow(flow_coarse, (features_q - features_qc).transpose(2, 1).contiguous().view(B, -1, H, W))
            
        if self.mode == 'pose':
            return None, q, t
            
        if self.mode == 'flow&pose':
            return flow, q, t
            
        if self.mode == 'flow->pose':
            flow_coarse = normalize_points(flow_coarse, K_s, scales_s)
            grid = create_normalized_grid((B, H, W), K_q, scales_q)
            flow_coarse = torch.cat((flow_coarse, grid), dim=1) #B x 4 x H x W
            q, t = self.pose_regressor(flow_coarse, K_q, K_s, scales_q, scales_s)
            return flow, q, t
