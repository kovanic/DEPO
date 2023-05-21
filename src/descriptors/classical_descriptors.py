import cv2 as cv
import numpy as np
from tqdm.auto import tqdm

def SIFT_detector(img, **params):
    sift = cv.SIFT_create(**params)
    key_points = sift.detect(img, None)
    return key_points

def ORB_detector(img, **params):
    orb = cv.ORB_create(**params)
    key_points = orb.detect(img, None)
    return key_points

def SIFT_descriptor(img, key_points, **params):
    sift = cv.SIFT_create(**params)
    key_points, descriptors = sift.compute(img, key_points)
    confidence = np.array([point.response for point in key_points])
    key_points = cv.KeyPoint.convert(key_points).round()
    return key_points, confidence, descriptors

def ORB_descriptor(img, key_points, **params):
    orb = cv.ORB_create(**params)
    key_points, descriptors = orb.compute(img, key_points)
    confidence = np.array([point.response for point in key_points])
    key_points = cv.KeyPoint.convert(key_points).round()
    return key_points, confidence, descriptors


def apply_modular_extractor_to_image(
    img: str,
    detector: callable,
    descriptor: callable,
    detector_params: dict = dict(),
    descriptor_params: dict = dict()
):
    """
    key points are returned in the format (x, y, c) ~ (width, height, confidence)
    """
    key_points = detector(img, **detector_params)
    key_points, confidence, descriptors = descriptor(img, key_points, **descriptor_params)
    return key_points, confidence[:, None], descriptors
