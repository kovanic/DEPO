import torch
from torch import nn
import torch.nn.functional as F
from torch.linalg import inv
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append('../../src')
from datasets.sun3d.loader import *

"""
This code is used for calculating covisibility scores and plotting
covisible region, projections.
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


def project_points_image2camera(uv, depth, K):
    """Project points from image plane to camera frame.
    For each (u,v) corresponding depth should be provided.

    :param uv: coordinates in image frame (B x 2 x N)
    :param depth: depth (B x N)
    :param K: calibration matrix (3 x 3)
    :return: pointcloud in camera coordinates (B x 3 x N)
    """
    uv1 = to_homogeneous(uv)
    xyz = inv(K) @ uv1 * depth
    return xyz


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
    return xyz


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
    :return: projection of pointcloud to image frame (B x 2 x N), corresponding depth
    """
    uv = to_cartesian(K @ xyz)
    return uv,  xyz[:, -1, :]


def points2image(uv_0, depth_0, T_0, T_1, K_0, K_1):
    """
    Project points from image frame of camera_0 to image frame of camera_1.

    :param uv_0: points in image frame 0 (B x 2 x N)
    :param depth: corresponding depth observed with camera_0 (B x N)
    :param T_0: camera_0 extrinsics matrix (B x 4 x 4)
    :param T_1: camera_1 extrinsics matrix (B x 4 x 4)
    :param K_0: camera_0 calibration matrix (B x 3 x 3)
    :param K_1: camera_1 calibration matrix (B x 3 x 3)
    """
    xyz_0 = project_points_image2camera(uv_0, depth_0, K_0)
    xyz_1 = camera2camera(xyz_0, T_0, T_1)
    uv_1, depth_01 = camera2image(xyz_1, K_1)
    return uv_1, depth_01


def image2image(depth_0, T_0, T_1, K_0, K_1):
    """Project image frame of camera_0 to image frame of camera_1.

    :param depth_0: depth observed with camera_0 (B x H_d0 x W_d0)
    :param T_0: camera_0 extrinsics matrix (B x 4 x 4)
    :param T_1: camera_1 extrinsics matrix (B x 4 x 4)
    :param K_0: camera_0 calibration matrix (B x 3 x 3)
    :param K_1: camera_1 calibration matrix (B x 3 x 3)
    """
    xyz_0 = image2camera(depth_0, K_0)
    xyz_1 = camera2camera(xyz_0, T_0, T_1)
    uv_1, depth_01 = camera2image(xyz_1, K_1)
    return uv_1, depth_01


def mask_for_outliers(img_shape, uv):
    """
    :param img_shape: (H, W)
    :param uv: coordinates of pixels (B x 2 x N)
    :return: bool mask (B x 2 x N) of whether the given coordinate is inside the image frame
    """
    H, W = img_shape
    mask = (uv[:, 1, :] >= 0) & (uv[:, 1, :] < H) & (uv[:, 0, :] >= 0) & (uv[:, 0, :] < W)
    return mask


def depth_close(depth_0, depth_1, thresh):
    mask = torch.abs(depth_0 - depth_1) < thresh
    return mask


def calculate_covisibility_scores_one_side(
    img_0, img_1,
    depth_0, depth_1,
    T_0, T_1,
    K_0, K_1,
    thresh=0.3,
    return_covisible_pixels=False
):
    """Calculation of covisibility scores - ratio of #pixels observed with both cameras to
    #valid pixels. Projection is from img_0 to img_1.

    :param img_0: (B x H_c0 x W_c0 x 3)
    :param img_1: (B x H_c1 x W_c1 x 3)
    :param depth_0: (B x H_d0 x W_d0)
    :param depth_1: (B x H_d1 x W_d1)
    :param T_0: camera_0 extrinsics matrix (B x 4 x 4)
    :param T_1: camera_1 extrinsics matrix (B x 4 x 4)
    :param K_0: camera_0 calibration matrix (B x 3 x 3)
    :param K_1: camera_1 calibration matrix (B x 3 x 3)
    :param thresh: maximum distance between original and projected depth from the second
    camera for the pixel to remain covisible.
    :return: if return_covisible_pixels=False then returns covisibility scores on batch
    else returns mask for covisible pixels and all filters
    """

    #In uv_01 we have Inf values for the pixels in which depth_0 == 0
    B, H, W = depth_0.size()
    uv_01, depth_01 = image2image(depth_0, T_0, T_1, K_0, K_1)
    uv_01 = uv_01.masked_fill_(uv_01.isinf(), -2.).round().long()
    proj_mask_01 = mask_for_outliers((H, W), uv_01)
    uv_01 = uv_01.masked_fill_(~proj_mask_01.unsqueeze(1), 0.0)


    depth_1_corr = torch.vstack([depth_1[i, uv_01[i, 1], uv_01[i, 0]] for i in range(B)])

    depth_valid_mask_0 = depth_0.view(B, -1) > 1e-20
    depth_valid_mask_1 = depth_1_corr > 1e-20
    depth_nearness_mask = depth_close(depth_1_corr, depth_01, thresh)

    n_of_valid_pixels = (depth_1 > 1e-20).sum(dim=(1, 2)) #depth_valid_mask_1.sum(dim=1)
    covisible_pixels = proj_mask_01 & depth_valid_mask_0 & depth_valid_mask_1 & depth_nearness_mask

    covisibility_values = covisible_pixels.sum(dim=1) / n_of_valid_pixels

    if return_covisible_pixels:
        return covisible_pixels, proj_mask_01, depth_valid_mask_0, depth_valid_mask_1, depth_nearness_mask
    return covisibility_values


def calculate_covisibility_scores(
    img_0, img_1,
    depth_0, depth_1,
    T_0, T_1,
    K_0, K_1,
    thresh=0.1
):
    covisibility_score_01 = calculate_covisibility_scores_one_side(
        img_0, img_1,
        depth_0, depth_1,
        T_0, T_1,
        K_0, K_1,
        thresh
    )
    covisibility_score_10 = calculate_covisibility_scores_one_side(
        img_1, img_0,
        depth_1, depth_0,
        T_1, T_0,
        K_1, K_0,
        thresh
    )
    return torch.minimum(covisibility_score_01, covisibility_score_10)


####Visualization###
def plot_covisible_region(
    i,
    img_0, img_1,
    depth_0, depth_1,
    T_0, T_1,
    K_0, K_1,
    thresh=0.3):
    """
    img_0,  img_1
    covisible_010, covisible_101
    """
    covisible_01, *_ = calculate_covisibility_scores_one_side(
        img_0, img_1,
        depth_0, depth_1,
        T_0, T_1,
        K_0, K_1,
        thresh=thresh,
        return_covisible_pixels=True
    )
    img_0_covisible = img_0[i].clone()
    img_0_covisible.view(-1, 3)[~covisible_01[i], :] = 0

    covisible_10, *_ = calculate_covisibility_scores_one_side(
        img_1, img_0,
        depth_1, depth_0,
        T_1, T_0,
        K_1, K_0,
        thresh=thresh,
        return_covisible_pixels=True
    )
    img_1_covisible = img_1[i].clone()
    img_1_covisible.view(-1, 3)[~covisible_10[i], :] = 0

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    ax[0][0].imshow(img_0[i])
    ax[0][0].set_title('Image 0')
    ax[0][1].imshow(img_1[i])
    ax[0][1].set_title('Image 1')
    ax[1][0].imshow(img_0_covisible)
    ax[1][0].set_title('Covisible pixels plotted on Image 0')
    ax[1][1].imshow(img_1_covisible)
    ax[1][1].set_title('Covisible pixels plotted on Image 1')
    for ax_ in ax.flat:
        ax_.set_axis_off()
    plt.tight_layout()
    # plt.savefig('./covisibility.pdf')
    plt.show()


def plot_projection(img_0, img_1, uv_1):
    """
    Plot colored projection of image frame of camera_0 to
    image frame of camera_1.
    :param img_0: rgb image, camera_0
    :param img_1: rgb image, camera_1
    :uv_1: corresponding pixels in img_1 (2 x N)
    """
    mask = mask_for_outliers(img_1.shape[:2], uv_1).squeeze()
    keep_idx = uv_1[:, :, mask].long().squeeze()
    colors = img_0.reshape(-1, 3)

    img0_proj_img1 = torch.zeros(img_1.shape).byte()
    img0_proj_img1[keep_idx[1], keep_idx[0], :] = colors[mask]
    masked_proj = img0_proj_img1.clone()

    no_projection = img0_proj_img1.sum(dim=2) == 0
    img0_proj_img1[no_projection] = img_1[no_projection, :]

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    ax[0][0].imshow(img_0)
    ax[0][0].set_title('Image 0')
    ax[0][1].imshow(img_1)
    ax[0][1].set_title('Image 1')
    ax[1][0].imshow(masked_proj)
    ax[1][0].set_title('Projection 0->1')
    ax[1][1].imshow(img0_proj_img1)
    ax[1][1].set_title('Projection 0->1 + Image 1')
    plt.tight_layout()
    plt.show()

def plot_pointcloud_for_pair(i, img_0, img_1, depth_0, depth_1, T_0, T_1, K_0, K_1):
    points0 = (to_cartesian(inv(T_0) @ to_homogeneous(image2camera(depth_0, K_0)))[i].T).numpy()
    points1 = (to_cartesian(inv(T_1) @ to_homogeneous(image2camera(depth_1, K_1)))[i].T).numpy()
    points = np.vstack((points0, points1))

    colors0 = img_0[i].reshape(-1, 3).numpy() / 255
    colors1 = img_1[i].reshape(-1, 3).numpy() / 255
    colors = np.vstack((colors0, colors1))

    cam_pose0 = o3d.geometry.TriangleMesh.create_coordinate_frame(0.3).transform(inv(T_0[i]))
    cam_pose1 = o3d.geometry.TriangleMesh.create_coordinate_frame(0.3).transform(inv(T_1[i]))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd, cam_pose0, cam_pose1])

    
def calculate_covisibility_on_batch(path, device, batch_size):

    annotations = pd.read_csv(path+'initial_annotations.csv', header=None)

    dataset = SUN3dDataset(
        path=path,
        annotations=annotations
        )

    dl = DataLoader(dataset, shuffle=False, batch_size=batch_size, pin_memory=True)

    covisibility_scores = torch.Tensor([]).to(device)
    for img_0, depth_0, img_1, depth_1, T_0, T_1, K in tqdm(dl, total=len(dl)):
        img_0 = img_0.to(device)
        img_1 = img_1.to(device)
        depth_0 = depth_0.to(device)
        depth_1 = depth_1.to(device)
        T_0 = T_0.to(device)
        T_1 = T_1.to(device)
        K = K.to(device)

        covisibility_scores = torch.cat([covisibility_scores,
        calculate_covisibility_scores(
            img_0, img_1,
            depth_0, depth_1,
            T_0, T_1,
            K, K,
            thresh=0.2
        )]
    )

    np.save(path+'covisibility_scores.npy', covisibility_scores.detach().cpu().numpy())