import numpy as np
import cv2 as cv
from tqdm import tqdm
import argparse
import os
import pickle
import pandas as pd
from multiprocessing import cpu_count, Pool
import json
import sys
sys.path.append('../../src')

from utils.metrics import calculate_metrics


class NNMatcherPair:
    """
    Implementation of Nearest Neighbours Matching approach for stereo pair using OpenCV.
    """
    def __init__(self, normtype, crosscheck=False, ratio=None):
        if normtype == 'hamming':
            self.normtype = cv.NORM_HAMMING
        else:
            self.normtype = cv.NORM_L2
        if ratio is None:
            self.matcher = cv.BFMatcher(self.normtype, crosscheck)
            self.mode = 'nn'
        else:
            self.ratio = ratio
            self.matcher = cv.BFMatcher(self.normtype, False)
            self.mode = 'nn+ratio_test'

    def match(self, kp_0, kp_1, desc_0, desc_1, c_0, c_1):
        if self.normtype == cv.NORM_HAMMING:
            desc_0, desc_1 = desc_0.astype(np.uint8), desc_1.astype(np.uint8)
        if self.mode == 'nn':
            matches = self.matcher.match(desc_0, desc_1)
            match_idx = np.array([[matches[i].queryIdx, matches[i].trainIdx] for i in range(len(matches))])
        else:
            matches_01 = self.matcher.knnMatch(desc_0, desc_1, k=2)
            matches_10 = self.matcher.knnMatch(desc_1, desc_0, k=2)
            confident_matches_01 = self.ratio_test(matches_01)
            confident_matches_10 = self.ratio_test(matches_10)
            match_idx_01 = [(confident_matches_01[i].queryIdx, confident_matches_01[i].trainIdx) for i in range(len(confident_matches_01))]
            match_idx_10 = [(confident_matches_10[i].trainIdx, confident_matches_10[i].queryIdx) for i in range(len(confident_matches_10))]
            match_idx = np.array(list(set(match_idx_01).intersection(match_idx_10)))
        if len(match_idx.shape) == 2:
            match_0, match_1 = kp_0[match_idx[:, 0]], kp_1[match_idx[:, 1]]
            c_0, c_1 = c_0[match_idx[:, 0]], c_1[match_idx[:, 1]]
            return match_0, match_1, c_0, c_1
        else:
            return np.array([]), np.array([]), np.array([]), np.array([])

    def ratio_test(self, matches):
        confident_matches = []
        for m1, m2 in matches:
            if m1.distance < self.ratio * m2.distance:
                confident_matches.append(m1)
        return confident_matches

class NNMatcher:
    def __init__(self,
                 dataset,
                 normtype='l2',
                 crosscheck=False,
                 ratio=None,
                 path_to_save='default'
                 ):
        self.dataset = dataset
        self.matcher = NNMatcherPair(normtype, crosscheck, ratio)
        self.n_of_pairs = len(self.dataset)
        self.path_to_save = path_to_save

    def match(self, **params):
        metrics = dict()
        for idx in tqdm(range(self.n_of_pairs)):
            pair_id, img_0_path, img_1_path, descriptor_0, descriptor_1, T_0, T_1, depth_0, depth_1, K_0, K_1, covisibility_score, R_distance, *_ = self.dataset[idx]
            desc_0, desc_1 = descriptor_0[:, 3:], descriptor_1[:, 3:]
            kp_0, kp_1 = descriptor_0[:, :2], descriptor_1[:, :2] # (x=width, y=height)
            match_0, match_1 = self.matcher.match(kp_0, kp_1, desc_0, desc_1) #N x 2, N x 2
            metrics[pair_id] = calculate_metrics(
                match_0.T.astype('int'), match_1.T.astype('int'),
                (480, 640), (480, 640),
                depth_0, depth_1,
                T_0, T_1, K_0, K_1,
                **params
                )
            metrics[pair_id].update({'covisibility_score': covisibility_score, 'R_distance_true': R_distance})

        with open(self.path_to_save, 'wb') as f:
            pickle.dump(metrics, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NN matching procedure.')
    parser.add_argument('path_to_configuration',
                         type=str,
                         help='Path to configuration file with parameters for matcher.')

    args = parser.parse_args()
    global configuration
    with open(args.path_to_configuration, 'r') as f:
        configuration = json.load(f)

    global method
    method = cv.RANSAC if configuration["method"] == 'RANSAC' else cv.USAC_MAGSAC

    def match_partial(tasks):
        dataset = DescriptorNNDataset(
            path_to_annotations=configuration["path_to_annotations"],
            path_to_descriptors=configuration["path_to_descriptors"],
            path_to_data=configuration["path_to_data"],
            indicies=tasks
            )
        matcher = NNMatcher(
            dataset=dataset,
            normtype=configuration["normtype"],
            crosscheck=configuration["crosscheck"],
            ratio=configuration["ratio"],
            path_to_save=configuration["path_to_save"]+str(tasks[0])+'.pickle'
            )
        matcher.match(
            epipolar_threshold=configuration["epipolar_threshold"],
            radius=configuration["radius"],
            covisibility_threshold=configuration["covisibility_threshold"],
            method=method,
            prob=configuration["prob"],
            threshold=configuration["threshold"],
            maxiters=configuration["maxiters"]
            )

    dataset_len = len(pd.read_csv(configuration["path_to_annotations"]))
    jobs = cpu_count() if configuration["jobs"] == -1 else configuration["jobs"]
    tasks = np.split(np.arange(dataset_len), np.linspace(0, dataset_len, jobs+1, dtype=int)[1:-1])

    pool = Pool(jobs)
    pool.map(match_partial, tasks)
    pool.close()
    pool.join()

    def concatenate_pickles(path):
        if os.path.isfile(path+'.pickle'):
            os.remove(path+'.pickle')
        group = path.split('/')[-1]
        path_to_files = '/'.join(path.split('/')[:-1])
        files = os.listdir(path_to_files)
        all_data = dict()
        for file in sorted(files):
            if group in file:
                with open(path_to_files+'/'+file, 'rb') as f:
                    all_data.update(pickle.load(f))
                os.remove(path_to_files+'/'+file)
        with open(path+'.pickle', 'wb') as f:
            pickle.dump(all_data, f)

    concatenate_pickles(configuration["path_to_save"])



# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser(description='NN matching procedure.')
#     parser.add_argument('pan',
#                          type=str,
#                          help='path to annotations')
#     parser.add_argument('pde',
#                          type=str,
#                          help='path to descriptors')
#     parser.add_argument('pda',
#                          type=str,
#                          help='path to data')
#     parser.add_argument('psa',
#                          type=str,
#                          help='path to save')
#     parser.add_argument('-n',
#                         '--norm',
#                         dest='norm',
#                         type=str,
#                         help='normtype')
#     parser.add_argument('-r',
#                         '--ratio',
#                         dest='ratio',
#                         type=float,
#                         help='normtype',
#                         default=None)
#     parser.add_argument('-crosscheck',
#                         action='store_true',
#                         help='whether to use mutual NN')
#     parser.add_argument('--jobs',
#                         help='number of cpu cores to use',
#                         type=int,
#                         default=-1)
#     args = parser.parse_args()
#
#     def match_partial(tasks):
#         dataset = DescriptorNNDataset(
#             path_to_annotations=args.pan,
#             path_to_descriptors=args.pde,
#             path_to_data=args.pda,
#             indicies=tasks
#             )
#         matcher = NNMatcher(
#             dataset=dataset,
#             normtype=args.norm,
#             crosscheck=args.crosscheck,
#             ratio=args.ratio,
#             path_to_save=args.psa+str(tasks[0])+'.pickle',
#             )
#         matcher.match()
#
#     dataset_len = len(pd.read_csv(args.pan))
#     jobs = cpu_count() if args.jobs == -1 else args.jobs
#     tasks = np.split(np.arange(dataset_len), np.linspace(0, dataset_len, jobs+1, dtype=int)[1:-1])
#
#     pool = Pool(jobs)
#     pool.map(match_partial, tasks)
#     pool.close()
#     pool.join()
#
#     def concatenate_pickles(path):
#         if os.path.isfile(path+'.pickle'):
#             os.remove(path+'.pickle')
#         group = path.split('/')[-1]
#         path_to_files = '/'.join(path.split('/')[:-1])
#         files = os.listdir(path_to_files)
#         all_data = dict()
#         for file in sorted(files):
#             if group in file:
#                 with open(path_to_files+'/'+file, 'rb') as f:
#                     all_data.update(pickle.load(f))
#                 os.remove(path_to_files+'/'+file)
#         with open(path+'.pickle', 'wb') as f:
#             pickle.dump(all_data, f)
#
#     concatenate_pickles(args.psa)
