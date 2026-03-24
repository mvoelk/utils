"""
SPDX-License-Identifier: MIT
Copyright © 2015 - 2026 Markus Völk
Code was taken from https://github.com/mvoelk/utils
"""

import tensorflow as tf
from keras import ops

eps = 1e-6


def norm(x, axis=-1):
    return ops.sqrt(ops.sum(ops.square(x), axis=axis) + eps)

def trafo(t, R):
    upper = tf.concat([R, tf.expand_dims(t,-1)], axis=-1)
    lower = tf.concat([tf.zeros_like(R[...,:1,:3]), tf.ones_like(R[...,:1,:1])], axis=-1)
    return tf.concat([upper, lower], axis=-2)

def hinv(T):
    Rt = tf.linalg.matrix_transpose(T[..., :3, :3])
    t = T[..., :3, 3:4]
    upper = tf.concat([Rt, -tf.matmul(Rt, t)], axis=-1)
    lower = tf.concat([tf.zeros_like(T[...,:1,:3]), tf.ones_like(T[...,:1,:1])], axis=-1)
    return tf.concat([upper, lower], axis=-2)

def transform_points(xyz, T):
    dtype = xyz.dtype
    xyz = tf.concat([xyz, tf.ones_like(xyz[...,:1])], axis=-1)
    xyz = tf.einsum('...ij,...nj->...ni', T, xyz)
    return tf.cast(xyz[...,:3], dtype)

def rotate_points(xyz, R):
    dtype = xyz.dtype
    xyz = tf.einsum('...ij,...nj->...ni', R, xyz)
    return tf.cast(xyz, dtype)

def gram_schmidt(R):
    """
    Gram-Schmidt orthogonalization for SO(3)
    """
    v1, v2 = R[...,0], R[...,1]

    e1 = tf.math.l2_normalize(v1, axis=-1)

    proj_v2_on_e1 = tf.reduce_sum(e1 * v2, axis=-1, keepdims=True) * e1
    u2 = v2 - proj_v2_on_e1
    e2 = tf.math.l2_normalize(u2, axis=-1)

    e3 = tf.linalg.cross(e1, e2)
    e3 = tf.math.l2_normalize(e3, axis=-1)

    Q = tf.stack([e1, e2, e3], axis=-1)
    return Q # Q from QR decomposition

def rot_regularizer(R):
    RtR = ops.matmul(ops.transpose(R,(0,2,1)), R)
    reg1 = ops.mean(ops.sqrt(ops.sum(ops.square(RtR-ops.eye(3)), axis=(-1,-2))+eps)) # orthogonality
    reg2 = ops.mean(ops.square(ops.det(R) - 1)) # determinant = +1
    return reg1 + reg2

def rotation_error(R1, R2):
    arg = ops.sum((R2-R1)**2, axis=(-1,-2))**0.5 / (2*2**0.5)
    arg = ops.clip(arg, -1+eps, 1-eps)
    return 2*ops.arcsin(arg)


