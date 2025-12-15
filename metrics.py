"""
SPDX-License-Identifier: MIT
Copyright © 2017 - 2025 Markus Völk
Code was taken from https://github.com/mvoelk/utils
"""

import numpy as np


def covariance_matrix(z):
    """Covariance of the features in a batch

    # Arguments
        z: shape (num_samples, num_features)
    """
    z = z - np.mean(z, axis=0)
    return np.sum( z[:,:,None] @ z[:,None,:], axis=0) / (z.shape[0] - 1)

def correlation_matrix(z):
    """Correlation of the features in a batch

    # Arguments
        z: shape (num_samples, num_features)
    """
    z = z - np.mean(z, axis=0)
    cov = np.sum( z[:,:,None] @ z[:,None,:], axis=0) / (z.shape[0] - 1)
    std = np.sqrt(np.diag(cov))
    std_outer = std[:,None] @ std[None,:]
    corr = cov / std_outer
    corr[std_outer==0] = 0  # handle zero-variance
    return corr

def similarity_matrix(z):
    """Cosine similarity between a batch of vectors

    # Arguments
        z: shape (num_samples, num_features)
    """
    zn = np.linalg.norm(z, axis=-1)
    return (z @ z.T) / (zn[:,None] @ zn[None,:])

def distance_matrix(z):
    """Euclidean distance between a batch of vectors

    # Arguments
        z: shape (num_samples, num_features)
    """
    return np.linalg.norm(z[:,None,:] - z[None,:,:], axis=-1)


def cosine_similarity(vectors1, vectors2):
    """Cosine similarity between batches of vectors or a batch and a vector

    # Arguments
        vectors1: shape (num_vectors, num_features)
        vectors2: shape (num_vectors, num_features) or (num_features)

    # Return
        shape (num_vectors)
    """
    vs, v = vectors1, vectors2
    vsn, vn = np.linalg.norm(vs, axis=-1), np.linalg.norm(v, axis=-1)
    return np.sum((vs*v), axis=-1) / (vsn*vn)

def euclidean_distance(vectors1, vectors2):
    """Euclidean distance between batches of vectors or a batch and a vector

    # Arguments
        vectors1: shape (num_vectors, num_features)
        vectors2: shape (num_vectors, num_features) or (num_features)

    # Return
        shape (num_vectors)
    """
    return np.linalg.norm(vectors1 - vectors2, axis=-1)

def mean_absolute_difference(vectors1, vectors2):
    """Mean absolute difference (MAD) between batches of vectors or a batch and a vector

    # Arguments
        vectors1: shape (num_vectors, num_features)
        vectors2: shape (num_vectors, num_features) or (num_features)

    # Return
        shape (num_vectors)
    """
    return np.mean(np.abs(vectors1 - vectors2), axis=-1)


def intersection_over_union(boxes, box, fix_points=True):
    """Intersection over union for a batch of axis-aligned bounding
    boxes with a given bounding box.

    # Arguments
        boxes: Batch of bounding boxes, array of shape (num_boxes, 4).
        box: Bounding box, array of shape (4).
            (x1, y1, x2, y2), x1 < x2, y1 < y2

    # Return
        iou: array of shape (num_boxes).
    """

    if fix_points:
        boxes = np.concatenate([np.minimum(boxes[:,0:2], boxes[:,2:4]), np.maximum(boxes[:,0:2], boxes[:,2:4])], axis=-1)
        box = np.concatenate([np.minimum(box[0:2], box[2:4]), np.maximum(box[0:2], box[2:4])], axis=-1)

    # intersection
    inter_tl = np.maximum(boxes[:,0:2], box[0:2])
    inter_br = np.minimum(boxes[:,2:4], box[2:4])
    inter_wh = np.maximum(inter_br - inter_tl, 0)
    inter = inter_wh[:,0] * inter_wh[:,1]

    # union
    box_a = (box[2] - box[0]) * (box[3] - box[1])
    boxes_a = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
    union = boxes_a + box_a - inter

    return inter / union


# legacy stuff
calc_similarity_matrix, calc_distance_matrix = similarity_matrix, distance_matrix
calc_cosine_similarity, calc_euclidean_distance = cosine_similarity, euclidean_distance
