import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
import os
import sys
sys.path.append('../../src')


class DescriptorNNDataset:
    """Create dataset, where each observation conists of detected keypoints and
    descriptors for two images and relative pose between corresponding cameras.
    """
    def __init__(self,
                 path_to_annotations,
                 path_to_descriptors,
                 path_to_data,
                 indicies=None):
        if indicies is None:
            self.annotations = pd.read_csv(path_to_annotations)
        else:
            self.annotations = pd.read_csv(path_to_annotations).iloc[indicies]
        self.path_to_descriptors = path_to_descriptors
        self.path_to_data = path_to_data
        sequences = [f for f in os.listdir(path_to_data) if os.path.isdir(path_to_data+f)]
        self.T = {seq_path: np.load(path_to_data+seq_path+'/extrinsics.npz')
                    for seq_path in sequences}
        self.K = {seq_path: read_intrinsics(path_to_data+seq_path+'/')
                    for seq_path in sequences}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        seq, img_0_path, depth_0_path, img_1_path, depth_1_path, covisibility_score, R_distance, *_ =  self.annotations.iloc[idx]
        pair_id = seq + img_0_path.split('.')[0] + img_1_path.split('.')[0]
        depth_0 = read_depth(self.path_to_data + seq + '/depth/' + depth_0_path)
        depth_1 = read_depth(self.path_to_data + seq + '/depth/' + depth_1_path)
        descriptor_0 = np.load(self.path_to_descriptors + seq + img_0_path.split('.')[0] + '.npy')
        descriptor_1 = np.load(self.path_to_descriptors + seq + img_1_path.split('.')[0] + '.npy')
        T_0, T_1 = self.T[seq][img_0_path], self.T[seq][img_1_path]
        K_0, K_1 = self.K[seq], self.K[seq]
        return pair_id, img_0_path, img_1_path, descriptor_0, descriptor_1, T_0, T_1, depth_0, depth_1, K_0, K_1, covisibility_score, R_distance


##################################Visualization#################################


def plot_matches(img_0, img_1, match_0, match_1, correct, ax, plot_connections=True, idx=None):
    if idx is None:
        idx = np.arange(len(match_1))
    img = np.hstack((img_0, img_1))
    match_1_ = match_1.copy()
    match_1_[:, 0] = match_1_[:, 0] + img_0.shape[1]
    if plot_connections:
        vertices = np.concatenate((match_0[idx, :], match_1_[idx, :]), 1).reshape(len(idx), 2, 2)
        if correct is None:
            colors='b'
        else:
            colors=['#00FF00' if pair else '#F23333' for pair in correct]
        lc = mc.LineCollection(vertices, linewidth=0.25, color=colors)
        ax.add_collection(lc)
    ax.imshow(img)
    ax.scatter(match_0[idx, 0], match_0[idx, 1], s=4, c='#82FFFA', edgecolors='#2DC341', linewidths=0.5)
    ax.scatter(match_1_[:, 0], match_1_[:, 1], s=4, c='#82FFFA', edgecolors='#2DC341', linewidths=0.5)
    ax.autoscale()
    ax.margins(0.1)
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,top=False,left=False,right=False,
        labelbottom=False,labeltop=False, labelleft=False, labelright=False
        )

    ax.set_aspect("auto")
