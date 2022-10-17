"""
SPDX-License-Identifier: MIT
Copyright © 2017 - 2022 Markus Völk
Code was taken from https://github.com/mvoelk/utils
"""

import numpy as np


def calc_similarity_matrix(z):
    """Cosine similarity between a batch of vectors

    # Arguments
        z: shape (num_samples, num_features)
    """
    zn = np.linalg.norm(z, axis=-1)
    return (z @ z.T) / (zn[:,None] @ zn[None,:])

def calc_distance_matrix(z):
    """Euclidean distance between a batch of vectors

    # Arguments
        z: shape (num_samples, num_features)
    """
    return np.linalg.norm(z[:,None,:] - z[None,:,:], axis=-1)


def calc_iou(boxes, box, fix_points=True):
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
