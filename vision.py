"""
SPDX-License-Identifier: MIT
Copyright © 2015 - 2024 Markus Völk
Code was taken from https://github.com/mvoelk/utils
"""


import numpy as np
import cv2


def xyz_to_image_orthographic(xyz, image_size, pixel_per_meter):
    '''Transforms from 3d space to image coordinates.

    # Arguments
        xyz: points in 3d space, shape (n, 3)
        image_size: shape (2)
        pixel_per_meter:

    # Return
        xy_img: shape (..., n, 2)
    '''
    w, h = image_size
    x_img = xyz[...,0] * pixel_per_meter + (w-1)/2
    y_img = xyz[...,1] * pixel_per_meter + (h-1)/2
    return np.stack([x_img, y_img], axis=-1)

#def xyz_to_image_perspective(xyz, K):
# TODO

def perspective_to_xyz(depth, K):
    '''Creates a point cloud from a depth map.

    # Arguments
        depth: depht map, shape (h, w)
        K: camera matrix, shape (3, 3)

    # Return
        xyz: points in 3d space, shape (h, w, 3)
    '''
    h, w = depth.shape
    z = np.float32(depth)
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    rx, ry = np.arange(w), np.arange(h)
    xi = np.repeat(rx[None,:],h,axis=0)
    yi = np.repeat(ry[:,None],w,axis=1)
    x = (xi - cx)*z / fx
    y = (yi - cy)*z / fy
    return np.stack([x,y,z], axis=-1)

def orthographic_to_xyz(depth, pixel_per_meter=300):
    """Creates a point cloud from an orthographic projection.

    # Arguments
        depth: depht map, shape (h, w)
        pixel_per_meter:

    # Return
        xyz: points in 3d space, shape (h, w, 3)
    """
    h, w = depth.shape
    z = np.float32(depth)
    rx, ry = np.arange(w), np.arange(h)
    xi = np.repeat(rx[None,:],h,axis=0)
    yi = np.repeat(ry[:,None],w,axis=1)
    x = (xi-(w-1)/2) / pixel_per_meter
    y = (yi-(h-1)/2) / pixel_per_meter
    return np.stack([x,y,z], axis=-1)


def xyz_to_orthographic(xyz, rgb=None, image_size=(512, 320), pixel_per_meter=300):
    """Creates an orthographic projection from a point cloud.

    # Arguments
        xyz: points in 3d space, shape (h, w, 3)
        rgb: color of points, shape (h, w, 3)
        image_size: shape (2)
        pixel_per_meter:

    # Return
        depth: orthograpic projekted depth map, shape (h, w)
        rgb: orthograpic projekted rgb image, shape (h, w, 3)
    """

    w, h = image_size

    xyz = np.reshape(xyz, (-1,3))
    idxs = np.argsort(-xyz[:,2])
    x, y, z = xyz[idxs].T

    x_img = np.int32(np.round( x * pixel_per_meter + (w-1)/2 ))
    y_img = np.int32(np.round( y * pixel_per_meter + (h-1)/2 ))

    m = np.logical_and(
        np.logical_and(x_img >= 0, x_img < w),
        np.logical_and(y_img >= 0, y_img < h))

    x_img, y_img = x_img[m], y_img[m]

    img = np.zeros((h,w), dtype='float32')
    img[y_img,x_img] = z[m]

    if rgb is not None:
        rgb = np.reshape(rgb, (-1,3))[idxs]
        rgb_ = np.zeros((h,w,3), dtype='float32')
        rgb_[y_img,x_img] = rgb[m]
        return img, rgb_

    return img

def perspective_to_orthographic(depth, rgb=None, K=np.eye(3), image_size=(512, 320), pixel_per_meter=300):
    """Creates an orthographic projection from a perspectiv depth map.

    # Arguments
        depth: depht map, shape (h, w)
        rgb: color of points, shape (h, w, 3)
        K: camera matrix, shape (3, 3)
        image_size: shape (2)
        pixel_per_meter:

    # Return
        depth: orthograpic projekted depth map, shape (h, w)
        rgb: orthograpic projekted rgb image, shape (h, w, 3)
    """
    xyz = perspective_to_xyz(depth, K)
    return xyz_to_orthographic(xyz, rgb, image_size=image_size, pixel_per_meter=pixel_per_meter)


def crop_and_scale_perspective(img, crop_xy=[0,0], crop_wh=[1000000,1000000], scale=1.0):
    '''Crops and Scales RGB and depth images or transform the camera matrix K.
    '''
    if img.shape == (3,3):
        K = np.copy(img)
        K[:2,2] -= crop_xy
        K[:2,:] *= scale
        return K
    else:
        (x,y), (w,h) = crop_xy, crop_wh
        img = img[y:y+h,x:x+w]
        h1,w1 = img.shape[:2]
        h2,w2 = scale*h1, scale*w1
        assert h2.is_integer(), w2.is_integer()
        if len(img.shape) == 2 or img.shape[2] == 1:
            # nearset interpolation for depth maps
            interpolation = cv2.INTER_NEAREST
        else:
            interpolation = cv2.INTER_CUBIC
        img = cv2.resize(img, (int(w2), int(h2)), interpolation=interpolation)
        return img

def normals_from_xyz(xyz, kernel_size=3, invalid_value=(0,0,0)):
    """Computes the normals of a structured point cloud
    
    # Arguments
        xyz: shape (h,w,3)
    """
    
    kernel = np.ones((kernel_size,kernel_size), dtype='float32')
    v = np.sum(np.abs(xyz), axis=-1)
    valid_mask = np.isfinite(v) & (v > 0)
    invalid_mask = cv2.filter2D(np.float32(~valid_mask), -1, kernel) != 0
    
    gy = cv2.Sobel(xyz, cv2.CV_64F, 1, 0, ksize=kernel_size)
    gx = cv2.Sobel(xyz, cv2.CV_64F, 0, 1, ksize=kernel_size)
    normals = np.cross(gx, gy, axis=-1)
    normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True)+1e-10)

    normals[invalid_mask] = invalid_value
    return normals

def center_of_mass(mask):
    """Calculates the center of mass of a mask"""
    moments = cv2.moments(np.uint8(mask>0))
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        return cx, cy
    else:
        return None

def bilinear_interpolate_points(img, xy):
    """
    # Arguments
        img: array of shape (h, w) or (h, w, c)
        xy: array of shape (k, 2)

    # Return
        fxy: array of shape (k) or (k, c)
    """

    xy = np.float32(xy).T
    x1, y1 = np.int32(xy)
    x2, y2 = x1+1, y1+1
    dx1, dy1 = xy%1
    dx2, dy2 = 1-dx1, 1-dy1

    if len(img.shape) == 3:
        dx1, dy1, dx2, dy2 = dx1[...,None], dy1[...,None], dx2[...,None], dy2[...,None]

    f11 = img[y1,x1]
    f12 = img[y2,x1]
    f21 = img[y1,x2]
    f22 = img[y2,x2]
    fxy = dx2 * ( f11*dy2 + f12*dy1 ) + dx1 * ( f21*dy2 + f22*dy1 )
    return fxy


def image_to_xyz_perspective(xy_img, z, K):
    '''Transforms from image coordinates to 3d space.

    # Arguments
        xy_img: shape (..., n, 2)
        z: depth, shape (..., n)
        K: camera matrix, shape (3, 3)

    # Return
        xyz: points in 3d space, shape (n, 3)
    '''
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    x = (xy_img[...,0] - cx) * z / fx
    y = (xy_img[...,1] - cy) * z / fy
    return np.stack([x,y,z], axis=-1)


def image_to_xyz_orthographic(xy_img, z, image_size, pixel_per_meter):
    '''Transforms from image coordinates to 3d space.

    # Arguments
        xy_img: shape (..., n, 2)
        z: depth, shape (..., n)
        image_size: shape (2)
        pixel_per_meter:

    # Return
        xyz: points in 3d space, shape (n, 3)
    '''
    w, h = image_size
    x = (xy_img[...,0]-(w-1)/2) / pixel_per_meter
    y = (xy_img[...,1]-(h-1)/2) / pixel_per_meter
    return np.stack([x,y,z], axis=-1)
