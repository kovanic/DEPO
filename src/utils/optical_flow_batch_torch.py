import torch
from torch import nn
import torch.nn.functional as F
from torch.linalg import inv


"""
This code is used for calculating of optical flow.
Commnet on notation: u, v are used for naming axis of image frames
u is corresponding to x(width) and v to y(height)
"""

def to_homogeneous(x):
    """Add row of 1 to second dimension of the tensor.

    :param x: tensor of shape (B=batchsize, K, N)
    """
    return torch.cat((x, torch.ones(x.size(0), 1, x.size(-1)).to(x.device)), dim=1)


def to_cartesian(x):
    """Transform from homogeneous to cartesian coordinates.

    :param x: tensor
    """
    return x[:, :-1, :] / x[:, -1, :].unsqueeze(1)


def create_coordinate_grid(img_shape):
    """
    :param img_shape: (batch_size, height, width)
    :return: coordinate grid of size (B x 2 x N), where N=height*width
    """
    B, H, W = img_shape
    u, v = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    uv = torch.cat((v[..., None], u[..., None]), dim=2).reshape(-1, 2).T
    uv = uv.repeat(B, 1, 1)
    return uv


def image2camera(depth, K):
    """
    Project image frame to camera coordinates.

    :param depth: depth image
    :param K: calibration matrix (3 x 3)
    :return: pointcloud in camera coordinates (B x 3 x N)
    """
    B, H, W = depth.shape
    uv = create_coordinate_grid((B, H, W)).to(depth.device)
    uv1 = to_homogeneous(uv).double()
    xyz = inv(K) @ uv1 * depth.view(B, 1, -1)
    return uv, xyz


def camera2camera(xyz, T_0, T_1):
    """
    Project pointcloud from coordinate system of the first camera to
    coordinate system of the second camera.

    :param xyz: pointcloud in first camera coordinates (B x 3 x N)
    :param T_0: extrinsics for the first camera (4 x 4)
    :param T_1: extrinsics for the second camera (4 x 4)
    :return: pointcloud in second camera coordinates (B x 3 x N)
    """
    xyz1 = to_homogeneous(xyz)
    xyz = to_cartesian(T_1 @ inv(T_0) @ xyz1)
    return xyz


def camera2image(xyz, K):
    """
    Project points from camera coordinates to image plane.

    :param xyz: pointcloud in camera coordinates (B x 3 x N)
    :param K: calibration matrix (3 x 3)
    :return: projection of pointcloud to image frame (B x 2 x N)
    """
    uv = to_cartesian(K @ xyz)
    return uv


def image2image(depth_0, T_0, T_1, K_0, K_1):
    """Project image frame of camera_0 to image frame of camera_1.

    :param depth_0: depth observed with camera_0 (B x H_d0 x W_d0)
    :param T_0: camera_0 extrinsics matrix (B x 4 x 4)
    :param T_1: camera_1 extrinsics matrix (B x 4 x 4)
    :param K_0: camera_0 calibration matrix (B x 3 x 3)
    :param K_1: camera_1 calibration matrix (B x 3 x 3)
    """
    uv_0, xyz_0 = image2camera(depth_0, K_0)
    xyz_1 = camera2camera(xyz_0, T_0, T_1)
    uv_1 = camera2image(xyz_1, K_1)
    return uv_0, uv_1


#CHEKED
def optical_flow(depth_0, T_0, T_1, K_0, K_1, mask=True):
    B, H, W = depth_0.shape
    uv_0, uv_1 = image2image(depth_0, T_0, T_1, K_0, K_1)
    flow = (uv_1 - uv_0)
    
    flow[:, 0, :]  = flow[:, 0, :] / W
    flow[:, 1, :]  = flow[:, 1, :] / H
    flow = flow.reshape(B, 2, H, W)
    
    if mask:
        depth_mask = depth_0 > 0
        flow_mask  = flow.abs()[:, 0, :, :] < 1
        total_mask = depth_mask & flow_mask
        return flow, total_mask.unsqueeze(1)
    
    return flow