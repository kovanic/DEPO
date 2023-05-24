import os.path as osp
import sys
file_name = osp.abspath(__file__)
dir_name = osp.dirname(file_name)
sys.path.append(dir_name)

import torch
from torch import nn
import torch.nn.functional as F

from quadtree_attention import QuadtreeAttention
from fourier_cross_attention import FourierCrossAttention
from twins_backbone import alt_gvt_large_partial

from focalnet_backbone import focalnet_base_partial
from pose_regressors import DensePoseRegressorV5



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


def make_fixed_pe(H, W, dim, scale=2*math.pi, temperature = 10_000):

    h = torch.linspace(0, 1, H)[:, None, None].repeat(1, W, dim)  # [0, scale]
    w = torch.linspace(0, 1, W)[None, :, None].repeat(H, 1, dim)

    dim_t = torch.arange(0, dim, 2).repeat_interleave(2)
    dim_t = temperature ** (dim_t / dim)

    h /= dim_t
    w /= dim_t

    h = torch.stack([h[:, :, 0::2].sin(), h[:, :, 1::2].cos()], dim=3).flatten(2)
    w = torch.stack([w[:, :, 0::2].sin(), w[:, :, 1::2].cos()], dim=3).flatten(2)

    pe = torch.cat((h, w), dim=2)
    return pe.permute(2, 0, 1)


class DEPO_v2(nn.Module):
    """
    :param mode: {'flow&pose', 'flow->pose', 'pose'}
    """
    def __init__(
        self, self_encoder, cross_encoder, pose_regressor,
        mode, hid_dim, hid_out_dim, upsample_factor=1,
        delta_layer='-', use_ln_in_decoder=False, add_abs_pos_enc=False,
        H=60, W=80):
        super(DEPO_v2, self).__init__()
         
        self.hid_dim = hid_dim
        self.hid_out_dim = hid_out_dim
        self.mode = mode
        self.upsample_factor = upsample_factor
        self.delta_layer = delta_layer
        self.add_abs_pos_enc = add_abs_pos_enc
        self.self_encoder = self_encoder
        self.cross_encoder = cross_encoder     
        
        self.geometry_decoder = nn.ModuleList()
        if delta_layer == 'cat':
            self.geometry_decoder.append(
                nn.Sequential(
                    nn.Linear(hid_dim * 2, hid_dim),
                    nn.LayerNorm(hid_dim) if use_ln_in_decoder else nn.Identity(),
                    nn.LeakyReLU(0.1)))
        
        self.geometry_decoder.append(
            nn.Sequential(
                nn.Linear(hid_dim, 1024),
                nn.LayerNorm(1024) if use_ln_in_decoder else nn.Identity(),
                nn.LeakyReLU(0.1),
                nn.Linear(1024, 1024),
                nn.LayerNorm(1024) if use_ln_in_decoder else nn.Identity(),
                nn.LeakyReLU(0.1),
                nn.Linear(1024, hid_dim),
                nn.LayerNorm(hid_dim) if use_ln_in_decoder else nn.Identity(),
                nn.LeakyReLU(0.1),
                nn.Linear(hid_dim, hid_out_dim),
                nn.LayerNorm(hid_out_dim) if use_ln_in_decoder else nn.Identity(),
                nn.LeakyReLU(0.1)))
        
        if add_abs_pos_enc:
            self.register_buffer("pe", make_fixed_pe(H, W, hid_out_dim // 2).unsqueeze(0))
        
        if self.mode in ('flow&pose', 'pose'):
            self.intrinsics_mlp = nn.Sequential(
                nn.Linear(20, hid_out_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(hid_out_dim, hid_out_dim * 2),
                nn.LeakyReLU(0.1),
                nn.Linear(hid_out_dim * 2, hid_out_dim * 2))
        
        if 'flow' in self.mode:
            self.flow_regressor = nn.Sequential(
                nn.Conv2d(hid_out_dim, hid_out_dim, 3, 1, 1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(hid_out_dim, hid_out_dim // 2, 1, 1, 0),
                nn.LeakyReLU(0.1),
                nn.Conv2d(hid_out_dim // 2, hid_out_dim // 2, 3, 1, 1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(hid_out_dim // 2, 2, 1, 1, 0),
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
        
        if len(features.size()) == 4: # For FocalNet output already has size B*2 x HW x hid_dim
            features = features.flatten(2).transpose(2, 1)
        features_q, features_s = features.split(B, dim=0) # B x HW x hid_dim
        
        #Apply cross-attention module
        features_qc = self.cross_encoder(features_q, features_s, H, W) # B x HW x hid_dim

        #Hidden geometry extraction
        if self.delta_layer == "-": 
            features_q_delta = features_q - features_qc
        if self.delta_layer == "r-": 
            features_q_delta = features_qc - features_q
        if self.delta_layer == "cat":
            features_q_delta = torch.cat([features_q, features_qc], dim=2)
            
        hidden_geometry = self.geometry_decoder(features_q_delta).transpose(2, 1).unflatten(2, (H, W)) # N x hid_dim x H x W

        if self.mode in ('pose', 'flow&pose'):
            calibration_vector = self.calibration_to_vector_(K_q, K_s, scales_q, scales_s)
            calibration_parameters = self.intrinsics_mlp(calibration_vector) # hid_out_dim*2
            mu, sigma = calibration_parameters[:, :self.hid_out_dim, None, None], torch.exp(calibration_parameters[:, self.hid_out_dim:, None, None])
            if self.add_abs_pos_enc:
                hidden_geometry = hidden_geometry + self.pe.repeat(B, 1, 1, 1)
            pose_regressor_input = (hidden_geometry - mu) / sigma

            q, t = self.pose_regressor(pose_regressor_input)
        
        if 'flow' in self.mode:
            flow_coarse = self.flow_regressor(hidden_geometry)
            flow = self.upsample_flow(flow_coarse, features_q_delta.transpose(2, 1).unflatten(2, (H, W)))
            
        if self.mode == 'pose':
            return None, q, t
            
        if self.mode == 'flow&pose':
            return flow, q, t
            
        if self.mode == 'flow->pose':
            flow_coarse = normalize_points(flow_coarse, K_s, scales_s)
            grid = create_normalized_grid((B, H, W), K_q, scales_q)
            flow_coarse = torch.cat((flow_coarse, grid), dim=1) #B x 4 x H x W
            q, t = self.pose_regressor(flow_coarse)
            return flow, q, t
            

############################Configurations############################
######################################################################

def A1():
    self_encoder = alt_gvt_large_partial(img_size=(480, 640))
    self_encoder.load_state_dict(torch.load(osp.join(dir_name, 'weights_external/pvt_large.pth')), strict=False)
    cross_encoder = QuadtreeAttention(dim=256, num_heads=8, topks=[16, 16, 8], scale=3)
    pose_regressor = DensePoseRegressorV5(128)
    return DEPO_v2(
        self_encoder=self_encoder,
        cross_encoder=cross_encoder,
        pose_regressor=pose_regressor,
        hid_dim=256,
        hid_out_dim=128,
        mode='flow&pose',
        upsample_factor=8
        delta_layer='-',
        use_ln_in_decoder=False,
        add_abs_pos_enc=False,
        H=60, W=80)


def A2():
    self_encoder = alt_gvt_large_partial(img_size=(480, 640))
    self_encoder.load_state_dict(torch.load(osp.join(dir_name, 'weights_external/pvt_large.pth')), strict=False)
    cross_encoder = QuadtreeAttention(dim=256, num_heads=8, topks=[16, 16, 8], scale=3)
    pose_regressor = DensePoseRegressorV5(128)
    return DEPO_v2(
        self_encoder=self_encoder,
        cross_encoder=cross_encoder,
        pose_regressor=pose_regressor,
        hid_dim=256,
        hid_out_dim=128,
        mode='flow&pose',
        upsample_factor=8
        delta_layer='-',
        use_ln_in_decoder=True,
        add_abs_pos_enc=False,
        H=60, W=80)


def A3():
    self_encoder = alt_gvt_large_partial(img_size=(480, 640))
    self_encoder.load_state_dict(torch.load(osp.join(dir_name, 'weights_external/pvt_large.pth')), strict=False)
    cross_encoder = QuadtreeAttention(dim=256, num_heads=8, topks=[16, 16, 8], scale=3)
    pose_regressor = DensePoseRegressorV5(128)
    return DEPO_v2(
        self_encoder=self_encoder,
        cross_encoder=cross_encoder,
        pose_regressor=pose_regressor,
        hid_dim=256,
        hid_out_dim=128,
        mode='flow&pose',
        upsample_factor=8
        delta_layer='r-',
        use_ln_in_decoder=False,
        add_abs_pos_enc=False,
        H=60, W=80)





