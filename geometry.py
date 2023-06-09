"""
SPDX-License-Identifier: MIT
Copyright © 2015 - 2022 Markus Völk
Code was taken from https://github.com/mvoelk/utils
"""


import numpy as np

eps = 1e-8


def rot2quat(R, xyzw=False):
    '''Convets a Rotation Matrix to a Unit Quaternion'''
    qr = np.sqrt(1+R[0,0]+R[1,1]+R[2,2]+eps) / 2
    # 24.83 µs
    qi = np.copysign(1/2 * np.sqrt(1+R[0,0]-R[1,1]-R[2,2]+eps), R[2,1]-R[1,2])
    qj = np.copysign(1/2 * np.sqrt(1-R[0,0]+R[1,1]-R[2,2]+eps), R[0,2]-R[2,0])
    qk = np.copysign(1/2 * np.sqrt(1-R[0,0]-R[1,1]+R[2,2]+eps), R[1,0]-R[0,1])
    # 14.53 µs, faster but larger error
    #qi = (R[2,1] - R[1,2]) / (4*qr)
    #qj = (R[0,2] - R[2,0]) / (4*qr)
    #qk = (R[1,0] - R[0,1]) / (4*qr)
    if xyzw:
        return np.array([qi, qj, qk, qr])
    else:
        return np.array([qr, qi, qj, qk])

def quat2rot(q, xyzw=False):
    '''Converts a Unit Quaternion to a Rotation Matrix'''
    if xyzw:
        qi, qj, qk, qr = q
    else:
        qr, qi, qj, qk = q
    R = np.array([
        [1-2*(qj**2+qk**2), 2*(qi*qj-qk*qr), 2*(qi*qk+qj*qr)],
        [2*(qi*qj+qk*qr), 1-2*(qi**2+qk**2), 2*(qj*qk-qi*qr)],
        [2*(qi*qk-qj*qr), 2*(qj*qk+qi*qr), 1-2*(qi**2+qj**2)],
    ])
    return R

def euler2rot(angles):
    '''Convert Euler Angles to Rotation Matrix (V-Rep convention)'''
    s1, s2, s3 = np.sin(angles)
    c1, c2, c3 = np.cos(angles)
    Rx = np.array([[1,0,0], [0,c1,-s1], [0,s1,c1]])
    Ry = np.array([[c2,0,s2], [0,1,0], [-s2,0,c2]])
    Rz = np.array([[c3,-s3,0], [s3,c3,0], [0,0,1]])
    return np.dot(Rx, np.dot(Ry, Rz))

def rot2euler(R, s=1):
    '''Convert Rotation Matrix to Euler Angles (V-Rep convention)'''
    a = np.arctan2(-s*R[1,2], s*R[2,2])
    b = np.arctan2(R[0,2], s*np.sqrt(1-R[0,2]*R[0,2]+eps))
    g = np.arctan2(-s*R[0,1], s*R[0,0])
    return np.array([a,b,g])

def euler2rot2(angles, axes='xyz', fixed=False):
    '''Converts arbitrary Euler Angles to Rotation Matrix

        V-Rep: 'xyz', False
        roll-pitch-yaw: 'xyz', True
    '''
    Rs = [[rotx, roty, rotz]['xyz'.index(c)](a) for a, c in zip(angles, axes.lower())]
    if fixed:
        Rs.reverse()
    return np.linalg.multi_dot(Rs)

def rot2euler2(R, axes='xyz', fixed=False, s=1):
    '''Converts a Rotation Matrix to Euler Angles

        V-Rep: 'xyz', False
        roll-pitch-yaw: 'xyz', True
    '''
    # for general solution see
    # https://github.com/ros/geometry/blob/noetic-devel/tf/src/tf/transformations.py#L1031

    eps = 1e-6
    if axes == 'xyz' and fixed:
        sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
        if sy < eps: # singular
            return np.array([
                np.arctan2(-R[1,2], R[1,1]),
                np.arctan2(-R[2,0], sy),
                0,
            ])
        else:
            return np.array([
                np.arctan2(R[2,1], R[2,2]),
                np.arctan2(-R[2,0], sy),
                np.arctan2(R[1,0], R[0,0])
            ])
        return np.array([x, y, z])
    elif axes == 'xyz' and not fixed:
        return np.array([
            np.arctan2(-s*R[1,2], s*R[2,2]),
            np.arctan2(R[0,2], s*np.sqrt(1-R[0,2]*R[0,2]+eps)),
            np.arctan2(-s*R[0,1], s*R[0,0]),
        ])
    else:
        return None

def pose2matrix(pose):
    x, y, z, a, b, c = pose
    T = np.array([[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]], dtype='float')
    T[:3,:3] = euler2rot([a, b, c])
    return T

def matrix2pos(T):
    x, y, z = T[:3,3]
    a, b, c = rot2euler(T[:3,:3])
    return np.array([x,y,z,a,b,c])

def rot2axisangle(R):
    arg = (R[0,0]+R[1,1]+R[2,2]-1)/2
    #arg = np.clip(arg, -1, 1)
    a = np.arccos(arg)
    if np.abs(a) > eps:
        x = R[2,1]-R[1,2]
        y = R[0,2]-R[2,0]
        z = R[1,0]-R[0,1]
        v = np.array([x,y,z])
        v = v / np.sqrt(np.dot(v,v))
        return v, a
    else:
        return np.array([0,0,0]), 0

def axisangle2rot(v, a):
    x, y, z = v
    s, c = np.sin(a), np.cos(a)
    t = 1.0 - c
    xs, ys, zs = x*s, y*s, z*s
    xyt, xzt, yzt = x*y*t, x*z*t, y*z*t
    R = np.array([
        [c+x*x*t, xyt-zs, xzt+ys],
        [xyt+zs, c+y*y*t, yzt-xs],
        [xzt-ys, yzt+xs, c+z*z*t],
    ])
    return R

def rot2vec(R):
    v, a = rot2axisangle(R)
    return v * a

def vec2rot(r):
    a = np.linalg.norm(r)
    v = r/a
    R = axisangle2rot(v, a)
    return R

def in2pi(r):
    '''Brings an angele in the intervall [-pi, pi]'''
    return r - np.floor(1/2*(r/np.pi+1))*2*np.pi # 1.35 µs
    #return np.mod(r+np.pi, 2*np.pi) - np.pi # 1.67 µs

def rotx(angle):
    s, c = np.sin(angle), np.cos(angle)
    return np.array([[1,0,0], [0,c,-s], [0,s,c]])

def roty(angle):
    s, c = np.sin(angle), np.cos(angle)
    return np.array([[c,0,s], [0,1,0], [-s,0,c]])

def rotz(angle):
    s, c = np.sin(angle), np.cos(angle)
    return np.array([[c,-s,0], [s,c,0], [0,0,1]])


def project_vector(v1, v2):
    """Decomposes vector v1 into an vector v1_proj_v2 parallel to v2 and an vector v1_orth_v2 orthogonal to vector v2.

    # Arguments
        v1: shape (..., 3)
        v2: shape (..., 3)

    # Return
        v1_proj_v2: component of v1 parallel to v2, shape (..., 3)
        v1_orth_v2: component of v1 orthogonal to v2, shape (..., 3)

    """
    v2_norm = np.linalg.norm(v2,axis=-1, keepdims=True)
    v1_proj_v2 = np.sum(v2*v1, axis=-1, keepdims=True) / v2_norm**2 * v2
    v1_orth_v2 = v1 - v1_proj_v2
    return v1_proj_v2, v1_orth_v2

def normalize_vector(v):
    """

    # Arguments
        v: shape (..., 3)

    # Return
        v_normalized: shape (..., 3)
    """
    return v/np.linalg.norm(v,axis=-1, keepdims=True)


def translation_error(t1, t2):
    '''Translation distance in form of the euclidean distance.

    # Arguments
        t1: array of shape (..., 3)
        t2: array of shape (..., 3)

    # Return
        array of shpae (...)
    '''
    return np.linalg.norm(t1-t2, axis=-1)

def rotation_error(R1, R2):
    '''Rotatation distance in form of the angle in angle-axis representation.

    # Arguments
        R1: array of shape (..., 3, 3)
        R2: array of shape (..., 3, 3)

    # Return
        array of shpae (...)
    '''
    # suffers from numeric issues at arccos(1)
    #R = np.matmul(R1, np.swapaxes(R2, -1, -2))
    #arg = (R[...,0,0] + R[...,1,1] + R[...,2,2] - 1.) / 2.
    ##arg = np.clip(arg, -1, 1)
    #arg = np.where(arg<1, arg, 2-arg)
    #arg = np.where(arg>-1, arg, -2-arg)
    #return np.arccos(arg)
    # much better :)
    arg = np.sum((R2-R1)**2, axis=(-1,-2))**0.5 / (2*2**0.5)
    #arg = np.clip(arg, -1, 1)
    arg = np.where(arg<1, arg, 2-arg)
    arg = np.where(arg>-1, arg, -2-arg)
    return 2*np.arcsin(arg)

def transformation_error(T1, T2):
    '''Translation and rotation distance of homogeneous matrices.

    # Arguments
        T1: array of shape (..., 4, 4)
        T2: array of shape (..., 4, 4)

    # Return
        array of shape (..., 2)
    '''
    if type(T1) is list: T1 = np.array(T1)
    if type(T2) is list: T2 = np.array(T2)
    return np.stack([
        translation_error(T1[...,:3,3], T2[...,:3,3]),
        rotation_error(T1[...,:3,:3], T2[...,:3,:3]),
    ], axis=-1)


def orthogonalize_matrix(R):
    '''Orthogonalizes a rotation matrix.

    # Arguments
        R: array of sahpe (..., 3, 3)

    # Notes
        z-axis is prioritized
        x-axis is orthogonal projected
        y-axis is constructed form z- and x-axis
    '''
    R = R / np.linalg.norm(R, axis=-2, keepdims=True)

    a_vec, b_vec = R[...,2], R[...,0]
    a_norm = np.linalg.norm(a_vec, axis=-1, keepdims=True)

    b_proj_a = np.sum(a_vec*b_vec, axis=-1, keepdims=True) / a_norm**2 * a_vec
    b_orth_a = b_vec - b_proj_a

    r1 = b_orth_a / np.linalg.norm(b_orth_a, axis=-1, keepdims=True)
    r2 = np.cross(a_vec, r1)
    r3 = a_vec
    R = np.stack([r1,r2,r3], axis=-1)

    return R

def hinv(T):
    '''Inverts a homogeneous matrix

    # Arguments
        T: shape (..., 4, 4)
    '''
    Ti = np.tile(np.eye(4), T.shape[:-2]+(1,1))
    Rt = np.swapaxes(T[...,:3,:3], -1, -2) # transpose
    Ti[...,:3,3] = - (Rt @ T[...,:3,None,3])[...,0]
    Ti[...,:3,:3] = Rt
    return Ti

def trafo(t=np.zeros(3), R=np.eye(3)):
    '''Buids a homogenious matrix

    # Arguments
        t: shape (..., 3)
        R: shape (..., 3, 3)
    '''
    T = np.tile(np.eye(4), R.shape[:-2]+(1,1))
    T[...,:3,:3] = R
    T[...,:3,3] = t
    return T

def to4x4(M):
    '''Translates homogeneous matrices form (...,3,4) to (...,4,4)'''
    z = np.zeros_like(M[...,:1,:])
    z[...,-1] = 1
    return np.concatenate([M,z], axis=-2)

def transform_points(T, xyz):
    '''
    # Arguments
        T: array of shape (4, 4)
        xyz: array of shape (..., 3)
    '''
    xyz = np.concatenate([xyz,np.ones_like(xyz[...,:1])], axis=-1)
    xyz = T @ xyz[...,None]
    return xyz[...,:3,0]


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
    xyz = depth_to_xyz(depth, K)
    return xyz_to_orthographic(xyz, rgb, image_size=image_size, pixel_per_meter=pixel_per_meter)

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


def hand_eye_calibration(A, B):
    """Hand-Eye Calibration following Shah 2013 

    Consider the equation AX = ZB of 4x4 homogeneous transformations, 
    where n valid A_i and B_i are known and X and Z are constant but 
    unknown. We solve for X and Z.

    # Arguments
        A: list of 4x4 homogeneous transformations
        B: list of 4x4 homogeneous transformations

    # Return
        X: 4x4 homogeneous transformation
        Z: 4x4 homogeneous transformation

    """

    r = range(len(A))

    K = np.sum([np.kron(B[i][:3,:3], A[i][:3,:3]) for i in r], axis=0)
    U, s, Vt = np.linalg.svd(K)

    # rotation (theorem 2.4)
    Vx = np.reshape(Vt[0,:], (3,3)).T
    Vz = np.reshape(U[:,0], (3,3)).T
    detVx, detVz = np.linalg.det(Vx), np.linalg.det(Vz)
    Rx = np.sign(detVx)*np.abs(detVx)**(-1/3) * Vx
    Rz = np.sign(detVz)*np.abs(detVz)**(-1/3) * Vz

    U, s, Vt = np.linalg.svd(Rx); Rx = np.dot(U, Vt)
    U, s, Vt = np.linalg.svd(Rz); Rz = np.dot(U, Vt)

    # translation
    I = np.eye(3)
    At = np.concatenate([np.concatenate([I,-A[i][:3,:3]], axis=1) for i in r], axis=0)
    bt = np.concatenate([A[i][:3,3] - np.dot(Rz, B[i][:3,3]) for i in r], axis=0)
    t = np.dot(np.linalg.pinv(At), bt)
    tx, tz = t[3:], t[:3]

    X = np.vstack([np.hstack([Rx, tx[:,None]]), [0,0,0,1]])
    Z = np.vstack([np.hstack([Rz, tz[:,None]]), [0,0,0,1]])

    return X, Z


# TODO: remove
depth_to_xyz = perspective_to_xyz
image_to_xyz = image_to_xyz_perspective


