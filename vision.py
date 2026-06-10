"""
SPDX-License-Identifier: MIT
Copyright © 2015 - 2025 Markus Völk
Code was taken from https://github.com/mvoelk/utils
"""


import numpy as np
import cv2


def new_kernel(kernel_size=3, shape=cv2.MORPH_ELLIPSE):
    return cv2.getStructuringElement(shape, (kernel_size, kernel_size))


def xyz_to_image_orthographic(xyz, image_size, pixel_per_meter):
    '''Transforms from 3d space to image coordinates.

    # Arguments
        xyz: points in 3d space, shape (..., 3)
        image_size: shape (2)
        pixel_per_meter:

    # Return
        xy_img: shape (..., 2)
    '''
    w, h = image_size
    x_img = xyz[...,0] * pixel_per_meter + (w-1)/2
    y_img = xyz[...,1] * pixel_per_meter + (h-1)/2
    return np.stack([x_img, y_img], axis=-1)

def xyz_to_image_perspective(xyz, K):
    '''Transforms from 3d space to image coordinates.

    # Arguments
        xyz: points in 3d space, shape (..., 3)
        K: camera matrix, shape (3, 3)

    # Return
        xy_img: shape (..., 2)
    '''
    xyz = K @ xyz[...,None]
    x, y, z = xyz[...,0,0], xyz[...,1,0], xyz[...,2,0]
    x_img, y_img = x/z, y/z
    return np.stack([x_img, y_img], axis=-1)


def perspective_to_xyz(depth, K):
    '''Creates a point cloud from a depth map.

    # Arguments
        depth: depth map, shape (h, w)
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
        depth: depth map, shape (h, w)
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
        depth: orthographic projected depth map, shape (h, w)
        rgb: orthographic projected rgb image, shape (h, w, 3)
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
    """Creates an orthographic projection from a perspective depth map.

    # Arguments
        depth: depth map, shape (h, w)
        rgb: color of points, shape (h, w, 3)
        K: camera matrix, shape (3, 3)
        image_size: shape (2)
        pixel_per_meter:

    # Return
        depth: orthographic projected depth map, shape (h, w)
        rgb: orthographic projected rgb image, shape (h, w, 3)
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
            # nearest interpolation for depth maps
            interpolation = cv2.INTER_NEAREST
        else:
            interpolation = cv2.INTER_CUBIC
        img = cv2.resize(img, (int(w2), int(h2)), interpolation=interpolation)
        return img


def normals_from_xyz(xyz, kernel_size=3, invalid_value=(0,0,0)):
    """Computes the normals of a structured point cloud

    We assume that the point cloud stems from a 3d camera, so the z component of the normals point to the origin.

    # Arguments
        xyz: shape (h,w,3)

    # Return
        normals: shape (h,w,3)
    """

    kernel = np.ones((kernel_size,kernel_size), dtype='float32')
    v = np.sum(np.abs(xyz), axis=-1)
    valid_mask = np.isfinite(v) & (v > 1e-8)
    invalid_mask = cv2.filter2D(np.float32(~valid_mask), -1, kernel) != 0

    gx = cv2.Sobel(xyz, cv2.CV_64F, 0, 1, ksize=kernel_size)
    gy = cv2.Sobel(xyz, cv2.CV_64F, 1, 0, ksize=kernel_size)
    normals = np.cross(gx, gy, axis=-1)
    normals = normals / np.maximum(1e-10, np.linalg.norm(normals, axis=-1, keepdims=True))
    normals[invalid_mask] = invalid_value

    return normals


def center_of_mass(mask):
    """Calculates the center of mass of a mask

    # Arguments
        mask: shape (h,w)

    # Return
        cx, cy: center of mass coordinates
    """
    moments = cv2.moments(np.uint8(mask>0))
    if moments['m00'] != 0:
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        return cx, cy
    else:
        return None

def oriented_bounding_box(mask, largest_contour=False):
    """Calculates the oriented bounding box for a mask

    # Arguments
        mask: shape (h,w)
        largest_contour: whether to use only the largest contour

    # Return
        (x, y): center of box
        (w, h): width is short side, height is long side
        angle: [0, 180], 0 is horizontal
        pts: shape (4,2), first point is left in angle direction
    """
    
    cnts, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return None

    contour = max(cnts, key=cv2.contourArea) if largest_contour else np.concatenate(cnts, axis=0)

    (x,y), (w,h), a = rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    angle = 180-a if w>h else 90-a

    w, h = min(w,h), max(w,h)

    if angle > 90:
        pts = [box[i] for i in (0,1,2,3)]
    else:
        pts = [box[i] for i in (1,2,3,0)]
    pts = np.int32(pts).tolist()

    return (x,y), (w,h), angle, pts

def axis_aligned_bounding_box(mask, largest_contour=False):
    """Calculates the axis-aligned bounding box for a mask

    # Arguments
        mask: shape (h,w)
        largest_contour: whether to use only the largest contour

    # Return
        (x_min, y_min, x_max, y_max): coordinates of bounding box
        (x, y): center of box
        (w, h): width and height of box
    """
    cnts, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return None

    contour = max(cnts, key=cv2.contourArea) if largest_contour else np.concatenate(cnts, axis=0)

    x, y, w, h = cv2.boundingRect(contour)
    x_min, y_min, x_max, y_max = x, y, x + w, y + h
    x, y = (x + w/2, y + h/2)

    return (x_min, y_min, x_max, y_max), (x, y), (w, h)


def draw_axis_aligned_bounding_box(img, xy=None, wh=None, pts=None, thickness=2):
    if xy is not None and wh is not None:
        (x, y), (w, h) = xy, wh
        x_min, y_min, x_max, y_max = x-w/2, y-h/2, x+w/2, y+h/2
        cv2.rectangle(img, (int(x_min-1),int(y_min-1)), (int(x_max+1),int(y_max+1)), (0,255,0), thickness)
    if pts is not None:
        x_min, y_min, x_max, y_max = pts
        cv2.rectangle(img, (int(x_min-1),int(y_min-1)), (int(x_max+1),int(y_max+1)), (0,0,255), thickness)
    return img

def draw_oriented_bounding_box(img, xy=None, wh=None, angle=None, pts=None, linewidth=2):
    if xy is not None and wh is not None and angle is not None:
        (x,y), (w,h) = xy, wh
        r = 0.5*min(w,h)
        a = angle / 180 * np.pi
        rect = (tuple(xy), tuple(wh), 90-angle)
        box = cv2.boxPoints(rect).astype(int)
        cv2.polylines(img, [box], True, (0,255,0), linewidth, lineType=cv2.LINE_AA)
        end = (int(x + r*np.cos(a)), int(y - r*np.sin(a)))
        cv2.line(img, (int(x), int(y)), end, (255,0,0), linewidth, lineType=cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), 2*linewidth, (0,0,255), -1, lineType=cv2.LINE_AA)
    if pts is not None:
        pts_i = np.int32(pts)
        cv2.polylines(img, [pts_i], True, (0,0,255), linewidth, lineType=cv2.LINE_AA)
        cv2.circle(img, tuple(pts_i[0]), 2*linewidth, (255,0,0), -1, lineType=cv2.LINE_AA)
    return img



def project_box_mask(T_box, box_size, K, image_size):
    """Projects a box or cuboid from the 3d space into an image mask

    # Arguments
        T_box: homogeneous transformation, center of box
        box_size: size in x-, y– and z-dimension
        K: camera matrix, shape (3,3)
        image_size: width, height

    # Return
        img: shape (h, w)
    """

    xyz = np.array([
        [-1., -1., -1.],
        [ 1., -1., -1.],
        [ 1.,  1., -1.],
        [-1.,  1., -1.],
        [-1., -1.,  1.],
        [ 1., -1.,  1.],
        [ 1.,  1.,  1.],
        [-1.,  1.,  1.],
    ]) * 0.5 * box_size

    # transform_points
    xyz = np.concatenate([xyz,np.ones_like(xyz[...,:1])], axis=-1)
    xyz = T_box @ xyz[...,None]
    xyz = np.ascontiguousarray(xyz[...,:3,0])

    rvec = tvec = np.zeros((3, 1), dtype='float32')

    xy, _ = cv2.projectPoints(xyz, rvec, tvec, K, None)
    xy = xy.astype(int)

    w, h = image_size
    mask = np.zeros((h,w), dtype='uint8')

    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7]
    ]

    for face in faces:
        pts = xy[face].reshape((-1, 1, 2))
        cv2.fillConvexPoly(mask, pts, 255)

    return mask


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


def find_local_maxima(img):
    """Finds the local maxima in an image.

    # Arguments
        img: shape (h,w)

    # Return
        xy: tuple with indices, each shape (n)
    """
    #kernel = np.ones((3,3), dtype='uint8')
    #kernel[1,1] = 0
    #dilated = cv2.dilate(img, kernel)
    #local_max = (img > dilated)
    
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(img, kernel)
    eroded = cv2.erode(img, kernel)
    local_max = (img == dilated) & (img > eroded)

    xy = np.where(local_max)
    return xy


def close_depth(depth, kernel_size=17):
    """Closes holes in depth maps using inpainting.
    
    # Arguments
        depth: uint16, shape (h,w)
    """
    m_depth = (depth > 0)
    inpainted = cv2.inpaint(depth, np.uint8(~m_depth), 1, cv2.INPAINT_TELEA)
    K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    m_closed = np.bool_(cv2.morphologyEx(np.uint8(m_depth), cv2.MORPH_CLOSE, K))
    inpainted[~m_closed] = 0
    return inpainted


def color_id_map(id_map, background_color=(0.1, 0.1, 0.1)):
    """
    # Arguments
        id_map: uint, shape (w,h)

    # Return
        img: float [0,1], shape (w,h,3)
    """
    img = np.zeros((*id_map.shape, 3), dtype='float32')
    for i in np.unique(id_map):
        if i == 0:
            c = np.array(background_color)
        else:
            c = np.random.random(3)
            a = 0.9
            c = a*c + (1-a)*np.ones_like(c)
        img[id_map==i] = c
    return img

def overlay_id_map(img, id_map, alpha=0.5):
    """
    # Arguments
        img: uint8 (h,w,3)
        id_map: uint (h,w)
        alpha: float [0, 1] strength of overlay

    # Return
        overlay: uint8 (h,w,3)
    """
    overlay = (1-alpha)*img + alpha*color_id_map(id_map)*255
    overlay = np.where(id_map[...,None], overlay, img)
    overlay = np.clip(overlay, 0, 255)
    return np.uint8(overlay)


def depth_as_rgb(img):
    vmin, vmax = np.nanmin(img), np.nanmax(img)
    if len(img.shape) == 2:
        img = img[:,:,None]
    if img.shape[2] == 1:
        img = np.tile(img, (1,1,3))
    return np.uint8((img-vmin)/(vmax-vmin)*255)


def read_rgb(file_path):
    return cv2.imread(file_path, cv2.IMREAD_UNCHANGED)[...,(2,1,0)]

def write_rgb(file_path, img):
    cv2.imwrite(file_path, np.uint8(img)[...,(2,1,0)], [int(cv2.IMWRITE_JPEG_QUALITY), 98])

def read_mask(file_path):
    return cv2.imread(file_path, cv2.IMREAD_UNCHANGED) > 0

def write_mask(file_path, mask):
    cv2.imwrite(file_path, np.uint8(mask>0)*255)


def tile_images(x):
    """
    # Arguments
        x: shape (m, n, h, w, ...)

    # Return
        image: shape (m*h, n*w, ...)
    """
    m, n, h, w = x.shape[:4]
    axes = (0,2,1,3) + tuple(range(4, x.ndim))
    return x.transpose(axes).reshape(m*h, n*w, *x.shape[4:])


# legacy
#depth_to_xyz = perspective_to_xyz
#image_to_xyz = image_to_xyz_perspective
