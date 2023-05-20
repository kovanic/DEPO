"""
This code is for estimation of matching qulaity metrics:
Mean Average Accuracy of reconstructed camera poses (mAA),
precision and recall.
"""

import numpy as np
from numpy.linalg import norm
import pickle
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import cv2 as cv
from .cameras_geometry_numpy import (to_homogeneous, to_cartesian, fundamental_matrix, essential_matrix, 
points2image, invert_calibration)


#CHECK
def rotation_angle_distance(R_0, R_1):
    """Given two rotations estimate angle distance (in degrees) between them."""
    return np.rad2deg(np.arccos((((R_0 @ R_1.T).trace() - 1) / 2).clip(-1, 1)))


#CHECK
def vector_angle_distance(t_0, t_1):
    """Given two vectors estimate angle distance (in degrees) between them."""
    n_0, n_1 = np.sqrt(t_0 @ t_0), np.sqrt(t_1 @ t_1)
    return np.rad2deg(np.arccos((t_0 @ t_1 / (n_0 * n_1)).clip(-1, 1)))


def rotation_angle_distance_batch(R0, R1):
    return np.rad2deg(np.arccos(((np.trace(R0 @ R1.transpose(0, 2, 1), axis1=1, axis2=2) - 1) / 2).clip(-1, 1)))


def vector_angle_distance_batch(t0, t1):
    n0, n1 = norm(t0, axis=1), norm(t1, axis=1)
    return np.rad2deg(np.arccos(((t0 * t1).sum(1) / (n0 * n1)).clip(-1, 1)))

#CHECK
def get_relative_pose(T_0, T_1):
    """Get realtive pose (0->1) from known extrinsics"""
    return T_1 @ np.linalg.inv(T_0)


#CHECK
def estimate_relative_pose(match_0, match_1, K_0, K_1=None, method=cv.RANSAC, prob=0.99999, threshold=1., maxiters=1000):
    """Robust estimation of relative pose of 1 camera with respect to 0 camera.
    match_0, macth_1: np.ndarray(N, 2)
    """
    #five-point algorithm requiremnet
    if match_0.shape[0] < 5:
        return None, None, None

    if K_1 is not None:
        match_0 = to_cartesian(invert_calibration(K_0) @ to_homogeneous(match_0.T, 0), 0).T
        match_1 = to_cartesian(invert_calibration(K_1) @ to_homogeneous(match_1.T, 0), 0).T
        K_0 = np.eye(3)

    E, inliers = cv.findEssentialMat(
        points1=match_0, points2=match_1,
        cameraMatrix=K_0, method=method,
        prob=prob, threshold=threshold,
        maxIters=maxiters, mask=True
        )

    try:
        _, R, t, inliers = cv.recoverPose(
            E=E, points1=match_0, points2=match_1, cameraMatrix=K_0, mask=inliers
            )
        return R, t, inliers
    except:
        return None, None, None


#CHECK
def epipolar_distance(match_0, match_1, T_01, K_0, K_1):
    """Calculate symmetric epipolar_distance.
    :param match_0: keypoints (x, y) in the first image frame (2 x N)
    :param match_1: matched keypoints in the second camera frame (2 x N)
    :param T_01: relative pose of the second camera to the first camera (4 x 4)
    :param K_{0, 1}: calibration matrice for corresponding camera (3 x 3)
    """
    #DIFFERENCE TO SUPERGLUE: they did the same but in camera frame. We did in image frame, so distance is in pixels
    f_0 = np.linalg.inv(K_0) @ to_homogeneous(match_0, 0) # 3 x N
    f_1 = np.linalg.inv(K_1) @ to_homogeneous(match_1, 0) # 3 x N
    E = essential_matrix(T_01)

    F_f_0 = E @ f_0 # 3 x N
    F_f_1 = E.T @ f_1 # 3 x N
        
    distance = (f_1 * F_f_0).sum(axis=0)**2 *\
               (1.0 / (F_f_0[0, :]**2 + F_f_0[1, :]**2) + 1.0 / (F_f_1[0, :]**2 + F_f_1[1, :]**2))
    return distance


def find_nearest_neighbour(values, queries):
    """Find 1 nearest neighbor among values for each query.
    :values, queries: np.ndarray(N, 3)
    :return: neighbor coordinates, distance to neighbour
    """
    nn = NearestNeighbors(n_neighbors=1, p=2, n_jobs=1)
    nn.fit(values)
    distances, idx = nn.kneighbors(queries , 1)
    return values[idx.squeeze(1)].T, distances.ravel()


#CHECK
def mask_for_outliers(img_shape, uv):
    """
    :param img_shape: (H, W)
    :param uv: coordinates of pixels (2 x N)
    :return: bool mask (2 x N) of whether the given coordinate is inside image frame
    """
    H, W = img_shape
    mask = (uv[1] >= 0) & (uv[1] < H) & (uv[0] >= 0) & (uv[0] < W)
    return mask


#CHECK
def depth_close(depth_0, depth_1, thresh):
    mask = np.abs(depth_0 - depth_1) < thresh
    return mask



def covisible_keypoints(uv_01, uv_10, img_0_shape, img_1_shape, depth_0, depth_1, depth_01, depth_10, covisibility_threshold):
    """Find which of keypoints projections are valid"""

    inliers_10 = mask_for_outliers(img_0_shape, uv_10)
    inliers_01 = mask_for_outliers(img_1_shape, uv_01)
    uv_10[:, ~inliers_10] = 0
    uv_01[:, ~inliers_01] = 0

    depth_0_corr = depth_0[uv_10[1, :], uv_10[0, :]]
    depth_1_corr = depth_1[uv_01[1, :], uv_01[0, :]]

    depth_valid_0 = depth_0_corr > 0
    depth_valid_1 = depth_1_corr > 0

    depth_close_10 = depth_close(depth_0_corr, depth_10, covisibility_threshold)
    depth_close_01 = depth_close(depth_1_corr, depth_01, covisibility_threshold)

    covisible_mask_10 = inliers_10 & depth_close_10 & depth_valid_0 & depth_valid_1
    covisible_mask_01 = inliers_01 & depth_close_01 & depth_valid_0 & depth_valid_1
    return covisible_mask_01, covisible_mask_10



def find_potential_matches(
    match_0, match_1, uv_01, uv_10, img_0_shape, image_1_shape, depth_0, depth_1, depth_01, depth_10, radius, covisibility_threshold
    ):
    covisible_01, covisible_10 = covisible_keypoints(
        uv_01, uv_10, img_0_shape, image_1_shape, depth_0, depth_1, depth_01, depth_10, covisibility_threshold)

    match_0 = match_0[:, covisible_01]
    uv_01 = uv_01[:, covisible_01]
    match_1 = match_1[:, covisible_10]
    uv_10 = uv_10[:, covisible_10]

    if (match_0.shape[1] == 0) or  (match_1.shape[1] == 0):
        return 0, 0

    match_10, d_10 = find_nearest_neighbour(match_0.T, uv_10.T) #on the 0 frame
    match_01, d_01 = find_nearest_neighbour(match_1.T, uv_01.T)

    match_0 = np.concatenate((match_10[:, d_10 < radius], match_1[:, d_10 < radius]), axis=0)
    match_1 = np.concatenate((match_0[:, d_01 < radius], match_01[:, d_01 < radius]), axis=0)
    potential_matches = array_intersection(match_0.T, match_1.T).T

    n_of_potential_matches = potential_matches.shape[1]
    return potential_matches, n_of_potential_matches



def check_match_correctness(
    match_0, match_1,
    img_0_shape, img_1_shape,
    depth_0, depth_1,
    T_0, T_1, K_0, K_1,
    epipolar_threshold, radius=10,
    covisibility_threshold=0.1):

    depth_0_corr = depth_0[match_0[1, :], match_0[0, :]]
    depth_1_corr = depth_1[match_1[1, :], match_1[0, :]]

    uv_01, depth_01 = points2image(match_0, depth_0_corr, T_0, T_1, K_0, K_1)
    uv_10, depth_10 = points2image(match_1, depth_1_corr, T_1, T_0, K_1, K_0)

    valid_depth_0 = depth_0_corr > 0
    valid_depth_1 = depth_1_corr > 0

    valid_projection_01 = np.linalg.norm(match_1 - uv_01, ord=2, axis=0) < radius
    valid_projection_10 = np.linalg.norm(match_0 - uv_10, ord=2, axis=0) < radius

    T_01 = get_relative_pose(T_0, T_1)
    valid_epipolar_distances = epipolar_distance(match_0, match_1, T_01, K_0, K_1) < epipolar_threshold

    valid_matches = (valid_projection_01 & valid_depth_0) | (valid_projection_10 & valid_depth_1) |\
                    (valid_epipolar_distances & ~valid_depth_0 & ~valid_depth_1)

    n_of_valid_matches = valid_matches.sum()
    potential_matches, n_of_potential_matches = find_potential_matches(
        match_0, match_1, uv_01, uv_10, img_0_shape, img_1_shape, depth_0, depth_1, depth_01, depth_10, radius,
        covisibility_threshold
        )
    if type(potential_matches) is np.ndarray:
        n_of_valid_among_potential_matches = len(array_intersection(potential_matches.T, np.concatenate((match_0, match_1), 0).T))
    else:
        n_of_valid_among_potential_matches = 0
    n_of_matches = match_0.shape[1]
    return n_of_matches, n_of_valid_matches, n_of_potential_matches, n_of_valid_among_potential_matches


#CHECK
def recall(n_of_valid_among_potential_matches, n_of_potential_matches):
    if n_of_potential_matches == 0:
        return None
    else:
        return n_of_valid_among_potential_matches / n_of_potential_matches


#CHECK
def precision(n_of_valid_matches, n_of_matches):
    if n_of_matches == 0:
        return None
    else:
        return n_of_valid_matches / n_of_matches


#CHECK
def mAA(R_distances, t_distances, thresh, N):
    t = np.linspace(0, thresh[0], N)[:, None]
    freq = ((R_distances <= t) & (t_distances <= thresh[1])).mean(axis=1)
    mAA_ = np.trapz(freq, t.ravel())
    return mAA_ / thresh[0]

def mAA_max(R_distances, t_angle_distances, thresh, N):
    distances = np.maximum(R_distances,  t_angle_distances)
    t = np.linspace(0, thresh, N)[:, None]
    freq = (distances <= t).mean(axis=1)
    mAA_ = np.trapz(freq, t.ravel())
    return mAA_ / thresh

def mAA_sep(distances, thresh, N):
    t = np.linspace(0, thresh, N)[:, None]
    freq = (distances <= t).mean(axis=1)
    mAA_ = np.trapz(freq, t.ravel())
    return mAA_ / thresh


#CHECK
def calculate_metrics(
    match_0, match_1,
    img_0_shape, img_1_shape,
    depth_0, depth_1,
    T_0, T_1, K_0, K_1,
    epipolar_threshold=1e-4, radius=10,
    covisibility_threshold=0.1,
    method=cv.RANSAC, prob=0.99999,
    threshold=3, maxiters=1000
    ):

    # Before robust estimation
    if (len(match_0.shape) == 1) or (match_0.shape[1] == 0):
        n_of_matches, n_of_valid_matches, n_of_potential_matches, n_of_valid_among_potential_matches = 0, 0, 0, 0
    else:
        n_of_matches, n_of_valid_matches, n_of_potential_matches, n_of_valid_among_potential_matches = check_match_correctness(
                match_0, match_1,
                img_0_shape, img_1_shape,
                depth_0, depth_1,
                T_0, T_1, K_0, K_1,
                epipolar_threshold, radius,
                covisibility_threshold
                )
    metrics = dict()
    metrics['N'] = n_of_matches
    metrics['precision'] = precision(n_of_valid_matches, n_of_matches)
    metrics['recall'] = recall(n_of_valid_among_potential_matches, n_of_potential_matches)

    #After robust estimation
    R, t, inliers = estimate_relative_pose(
        match_0.T, match_1.T, K_0, K_1=None, method=method, prob=prob, threshold=threshold, maxiters=maxiters
        )
    if R is None:
        metrics['R_distance'], metrics['t_distance'], metrics['N_after'], metrics['precision_after'], metrics['recall_after'] =\
        None, None, None, None, None
        return metrics
    else:
        T_01 = get_relative_pose(T_0, T_1)
        R_gt, t_gt = T_01[:3, :3],  T_01[:3, 3]
        metrics['R_distance'] = rotation_angle_distance(R_gt, R)
        metrics['t_distance'] = vector_angle_distance(t_gt, t.ravel())

        match_0 = match_0[:, inliers.astype(bool).ravel()]
        match_1 = match_1[:, inliers.astype(bool).ravel()]

        if (len(match_0.shape) == 1) or (match_0.shape[1] == 0):
            n_of_matches, n_of_valid_matches, n_of_potential_matches, n_of_valid_among_potential_matches = 0, 0, 0, 0
        else:
            n_of_matches, n_of_valid_matches, n_of_potential_matches, n_of_valid_among_potential_matches = check_match_correctness(
                    match_0, match_1,
                    img_0_shape, img_1_shape,
                    depth_0, depth_1,
                    T_0, T_1, K_0, K_1,
                    epipolar_threshold, radius,
                    covisibility_threshold
                    )

        metrics['N_after'] = n_of_matches
        metrics['precision_after'] = precision(n_of_valid_matches, n_of_matches)
        metrics['recall_after'] = recall(n_of_valid_among_potential_matches, n_of_potential_matches)
        return metrics

#CHECK
def calculate_metrics_simple(
    match_0, match_1,
    T_0, T_1, K_0, K_1,
    method=cv.RANSAC, prob=0.99999,
    threshold=3, maxiters=10000
    ):
    metrics = {}
    R, t, inliers = estimate_relative_pose(
        match_0.T, match_1.T, K_0, K_1=None, method=method, prob=prob, threshold=threshold, maxiters=maxiters
        )

    if R is None:
        metrics['R_distance'], metrics['t_distance'], metrics['N_after'] = None, None, None
        return metrics
    else:
        T_01 = get_relative_pose(T_0, T_1)
        R_gt, t_gt = T_01[:3, :3],  T_01[:3, 3]
        metrics['R_distance'] = rotation_angle_distance(R_gt, R)
        metrics['t_distance'] = vector_angle_distance(t_gt, t.ravel())
        metrics['N_after'] = inliers.sum()
        return metrics
    

#CHECK
def array_intersection(x0, x1):
    """Intersect two np.ndarrays.
    https://stackoverflow.com/a/8317403

    x0: np.ndarray(N0, K)
    x1: np.ndarray(N1, K)
    """
    nrows, ncols = x0.shape
    dtype = {'names':['f{}'.format(i) for i in range(ncols)],
       'formats':ncols * [x0.dtype]}
    intersection = np.intersect1d(x0.view(dtype), x1.view(dtype))
    return intersection.view(x1.dtype).reshape(-1, ncols)


#CHECK
def read_metrics(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return pd.DataFrame.from_dict(data, orient='index')
