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


def image2camera(depth, K):
    """Project image frame to camera coordinates.

    :param depth: depth image
    :param K: calibration matrix (3 x 3)
    :return: pointcloud in camera coordinates (3 x N)
    """
    uv = create_coordinate_grid(depth.shape)
    xyz = project_points_image2camera(uv, depth.reshape(1, -1), K)
    return xyz, uv


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


def image2image(depth_0, T_0, T_1, K_0, K_1):
    """Project image frame of camera_0 to image frame of camera_1.

    :param depth_0: depth observed with camera_0 (H_d0 x W_d0)
    :param T_0: camera_0 extrinsics matrix (4 x 4)
    :param T_1: camera_1 extrinsics matrix (4 x 4)
    :param K_0: camera_0 calibration matrix (3 x 3)
    :param K_1: camera_1 calibration matrix (3 x 3)
    :return: corrseponding pixels in camera_1 and depth (projected) in camera_1
    """
    xyz_0, uv_0 = image2camera(depth_0, K_0)
    xyz_1 = camera2camera(xyz_0, T_0, T_1)
    uv_1, depth_01 = camera2image(xyz_1, K_1)
    return uv_0, uv_1


def optical_flow(depth_0, T_0, T_1, K_0, K_1, mask=True, normalize=False):
    """Get optical flow from image_0 to image_1.

    :param depth_0: depth observed with camera_0 (H x W)
    :param T_0: camera_0 extrinsics matrix (4 x 4)
    :param T_1: camera_1 extrinsics matrix (4 x 4)
    :param K_0: camera_0 calibration matrix (3 x 3)
    :param K_1: camera_1 calibration matrix (3 x 3)
    :return: flow_0to1: (H x W), 
             mask: values out of image_1 boundaries & no depth_0 available,  (H x W)
    """

    H, W = depth_0.shape
    uv_0, uv_1 = image2image(depth_0, T_0, T_1, K_0, K_1)
    flow = (uv_1 - uv_0)
    flow = flow.reshape(2, H, W)
    
    if mask:
        depth_mask = depth_0 > 0
        coo_0 = uv_0.reshape(2, H, W)
        coo_1 = coo_0 + flow
        covisibility_mask = (coo_1[0, :, :] < W-1) & (coo_1[0, :, :] >= 0) & (coo_1[1, :, :] < H-1) & (coo_1[1, :, :] >= 0)
        total_mask = depth_mask & covisibility_mask
    
    if normalize:
        flow[0, :]  = flow[0, :] / W
        flow[1, :]  = flow[1, :] / H
    
    if mask:
        return flow, total_mask[None, :, :]
    else:
        return flow, None