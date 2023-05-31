import numpy as np

def to_homogeneous(coo, axis):
    """Homegenize coordinates.

    :param coo: cartesian coordinates
    :return: homogenized coordinates
    """
    if axis == 0:
        return np.concatenate((coo, np.ones((1, coo.shape[1]))), axis=0)
    elif axis == 1:
        return np.concatenate((coo, np.ones((coo.shape[0], 1))), axis=1)


def to_cartesian(coo, axis):
    """Dehomegenize coordinates.

    :param coo: homogeneous coordinates
    :return: dehomogenized coordinates
    """
    if axis == 0:
        return coo[:-1, :] / coo[-1, :]
    elif axis == 1:
        return coo[:, :-1] / coo[:, -1, None]


def invert_calibration(K):
    """Invert calibrarion matrice."""
    return np.array([[1/K[0, 0], 0        ,-K[0, 2]/K[0, 0]],
                     [0        , 1/K[1, 1],-K[1, 2]/K[1, 1]],
                     [0        , 0        , 1             ]])


def create_coordinate_grid(img_shape):
    """
    :param img_shape: (height, width)
    :return: coordinate grid of size (2 x N), where N=height*width
    """
    H, W = img_shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    grid = np.concatenate((u[..., None], v[..., None]), axis=-1).reshape(-1, 2).T
    return grid


def project_points_image2camera(uv, depth, K):
    """Project points from image plane to camera frame.
    For each (u,v) corresponding depth should be provided.

    :param uv: coordinates in image frame (2 x N)
    :param depth: depth (1 x N)
    :param K: calibration matrix (3 x 3)
    :return: pointcloud in camera coordinates (3 x N)
    """
    uv1 = to_homogeneous(uv, 0)
    xyz = depth * (invert_calibration(K) @ uv1)
    return xyz


def image2camera(depth, K):
    """Project image frame to camera coordinates.

    :param depth: depth image
    :param K: calibration matrix (3 x 3)
    :return: pointcloud in camera coordinates (3 x N)
    """
    uv = create_coordinate_grid(depth.shape)
    xyz = project_points_image2camera(uv, depth.reshape(1, -1), K)
    return xyz


def camera2camera(xyz, T_0, T_1):
    """Project pointcloud from coordinate system of the first camera to
    coordinate system of the second camera.

    :param xyz: pointcloud in first camera coordinates (3 x N)
    :param T_0: extrinsics for the first camera (4 x 4)
    :param T_1: extrinsics for the second camera (4 x 4)
    :return: pointcloud in second camera coordinates (3 x N)
    """
    xyz1 = to_homogeneous(xyz, 0)
    xyz = to_cartesian(T_1 @ np.linalg.inv(T_0) @ xyz1, 0)
    return xyz


def camera2image(xyz, K):
    """Project points from camera coordinates to image plane.

    :param xyz: pointcloud in camera coordinates (3 x N)
    :param K: calibration matrix (3 x 3)
    :return: projection of pointcloud to image frame (2 x N), corresponding depth values
    """
    uv = to_cartesian(K @ xyz, 0)
    return uv,  xyz[2, :]


def points2image(uv_0, depth_0, T_0, T_1, K_0, K_1):
    """Project points from image frame of camera_0 to image frame of camera_1.

    :param uv_0: points in image frame 0 (2 x N)
    :param depth: corresponding depth observed with camera_0 (N)
    :param T_0: camera_0 extrinsics matrix (4 x 4)
    :param T_1: camera_1 extrinsics matrix (4 x 4)
    :param K_0: camera_0 calibration matrix (3 x 3)
    :param K_1: camera_1 calibration matrix (3 x 3)
    """
    xyz_0 = project_points_image2camera(uv_0, depth_0, K_0)
    xyz_1 = camera2camera(xyz_0, T_0, T_1)
    uv_1, depth_01 = camera2image(xyz_1, K_1)
    return uv_1.round().astype('int'), depth_01


def image2image(depth_0, T_0, T_1, K_0, K_1):
    """Project image frame of camera_0 to image frame of camera_1.

    :param depth_0: depth observed with camera_0 (H_d0 x W_d0)
    :param T_0: camera_0 extrinsics matrix (4 x 4)
    :param T_1: camera_1 extrinsics matrix (4 x 4)
    :param K_0: camera_0 calibration matrix (3 x 3)
    :param K_1: camera_1 calibration matrix (3 x 3)
    :return: corrseponding pixels in camera_1 and depth (projected) in camera_1
    """
    xyz_0 = image2camera(depth_0, K_0)
    xyz_1 = camera2camera(xyz_0, T_0, T_1)
    uv_1, depth_01 = camera2image(xyz_1, K_1)
    return uv_1.round().astype('int'), depth_01


def essential_matrix(T):
    """Get essential matrix from relative pose."""
    t0, t1, t2 = T[:3, -1]
    t_skew = np.array([[0, -t2, t1],
                       [t2, 0, -t0],
                       [-t1, t0, 0]])
    return t_skew @ T[:3, :3]


def fundamental_matrix(T_01, K_0, K_1):
    """Get fundamental matrix form relative pose and calibration matrices."""
    return invert_calibration(K_1).T @ essential_matrix(T_01) @ invert_calibration(K_0)


def extrinsics2camera_pose(
    extrinsics: np.ndarray
    ) -> np.ndarray:
    """Get the camera poses in world coordinates.
    :param extrinsics: (#frames, 3, 4) array of extrinsic
    camera matrices. They specify transformations of world
    points to camera coordinates.
    https://ksimek.github.io/2012/08/22/extrinsic/
    :return: array (#frames, 3, 4)
    """
    Rc = extrinsics[:, :, :3].transpose(0, 2, 1)
    C = -Rc @ np.expand_dims(extrinsics[:, :, 3], 2)
    poses = np.concatenate((Rc, C), axis=2)
    return poses

####################################################################################################################################s

# def pointcloud_from_depth(
#     img_depth: np.ndarray,
#     K: np.ndarray
# ) -> np.ndarray:
#     """Transform points from image frame to camera frame [u, v, 1] * z @ (K^-1).T
#     :param img_depth: depth image
#     :param K: undistorted calibration matrix [3 x 3]
#     :return 3d pointcloud in camera coordinate system [N x 3]:
# #     """
#     shape = img_depth.shape[::-1]
#
#     grid_x, grid_y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
#     grid = np.concatenate((grid_x[..., None], grid_y[..., None]), axis=-1)
#
#     norm_grid = to_homogeneous(grid.reshape(-1, 2), 1) @ np.linalg.inv(K).T
#     pointcloud = norm_grid * np.expand_dims(img_depth.reshape(-1), axis=-1)
#     return pointcloud

import open3d as o3d

def geometries_for_one_frame(img, depth, T, K):
    #T w2c
    points = to_cartesian(np.linalg.inv(T) @ to_homogeneous(image2camera(depth, K), axis=0), axis=0).T
    colors = img.reshape(-1, 3) / 255
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    pose = o3d.geometry.TriangleMesh.create_coordinate_frame(0.3).transform(np.linalg.inv(T))

    return [pcd, pose]