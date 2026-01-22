"""
SPDX-License-Identifier: MIT
Copyright © 2015 - 2025 Markus Völk
Code was taken from https://github.com/mvoelk/utils
"""


import numpy as np

seqs = ('xyz', 'xyx', 'xzy', 'xzx', 'yzx', 'yzy', 'yxz', 'yxy', 'zxy', 'zxz', 'zyx', 'zyz')


def skew(v):
    x, y, z = v
    S = np.array([[ 0, -z,  y],
                  [ z,  0, -x],
                  [-y,  x,  0]], dtype=float)
    return S

def unskew(S):
    v = np.array([S[2,1], S[0,2], S[1,0]])
    return v


def rot2quat(R, xyzw=False):
    '''Convets a Rotation Matrix to a Unit Quaternion'''
    qr = 0.5 * np.sqrt(max(1+R[0,0]+R[1,1]+R[2,2],0.0))
    # 24.83 µs
    qi = np.copysign(0.5 * np.sqrt(max(1+R[0,0]-R[1,1]-R[2,2],0.0)), R[2,1]-R[1,2])
    qj = np.copysign(0.5 * np.sqrt(max(1-R[0,0]+R[1,1]-R[2,2],0.0)), R[0,2]-R[2,0])
    qk = np.copysign(0.5 * np.sqrt(max(1-R[0,0]-R[1,1]+R[2,2],0.0)), R[1,0]-R[0,1])
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

def euler2rot(angles, axes='xyz', fixed=False):
    '''Converts arbitrary Euler Angles to Rotation Matrix

        V-Rep: 'xyz' False
        Roll-Pitch-Yaw, ROS: 'xyz' True
        Yaw-Pitch-Roll: 'zyx' False
    '''
    Rs = [[rotx, roty, rotz]['xyz'.index(c)](a) for a, c in zip(angles, axes.lower())]
    if fixed:
        Rs.reverse()
    return np.linalg.multi_dot(Rs)

def rot2euler(R, axes='xyz', fixed=False, s=1):
    '''Converts a Rotation Matrix to Euler Angles

        V-Rep: 'xyz' False
        Roll-Pitch-Yaw, ROS: 'xyz' True
        Yaw-Pitch-Roll: 'zyx' False
    '''
    # for general solution see
    # https://github.com/ros/geometry/blob/noetic-devel/tf/src/tf/transformations.py#L1031

    eps = 1e-6
    if axes == 'xyz':
        if fixed:
            sy = np.sqrt(max(R[0,0]*R[0,0]+R[1,0]*R[1,0], 0.0))
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
        else:
            return np.array([
                np.arctan2(-s*R[1,2], s*R[2,2]),
                np.arctan2(R[0,2], s*np.sqrt(max(1-R[0,2]*R[0,2], 0.0))),
                np.arctan2(-s*R[0,1], s*R[0,0]),
            ])
    else:
        return None

def rot2euler_sp(R, axes='xyz', fixed=True):
    from scipy.spatial.transform import Rotation
    seq = axes.lower() if fixed else axes.upper()
    return Rotation.from_matrix(R).as_euler(seq, degrees=False)

def rot2axisangle(R):
    eps = 1e-6
    arg = 0.5 * (R[0,0]+R[1,1]+R[2,2] - 1)
    arg = np.clip(arg, -1, 1)
    a = np.arccos(arg)
    if a < eps:
        v = np.array([1,0,0])
    elif np.pi-a < eps:
        # for angle = pi, axis is the eigenvector corresponding to eigenvalue 1
        v = np.linalg.svd(R - np.eye(3))[0][:,-1]
    else:
        v = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
        v = v / np.linalg.norm(v)
    return np.array(v, dtype=R.dtype), np.float64(a)

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
    '''Brings an angles in the interval [-pi, pi]'''
    return r - np.floor(0.5*(r/np.pi+1))*2*np.pi # 1.35 µs
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

def random_quat():
    u, v, w = np.random.uniform(0, 1, size=3)
    r1 = np.sqrt(1.0 - u)
    r2 = np.sqrt(u)
    t1 = 2*np.pi * v
    t2 = 2*np.pi * w
    return np.array([np.cos(t2)*r2, np.sin(t1)*r1, np.cos(t1)*r1, np.sin(t2)*r2])

def random_rot():
    return quat2rot(random_quat())

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
        array of shape (...)
    '''
    return np.linalg.norm(t1-t2, axis=-1)

def rotation_error(R1, R2):
    '''Rotatation distance in form of the angle in angle-axis representation.

    # Arguments
        R1: array of shape (..., 3, 3)
        R2: array of shape (..., 3, 3)

    # Return
        array of shape (...)
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
        R: array of shape (..., 3, 3)

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
    '''Builds a homogeneous matrix

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

def print_rot_info(R):
    '''Prints some properties of a rotation matrix.
        R@R.T                 should be I
        norm(R, axis=-2)      should be all 1
        norm(R, axis=-1)      should be all 1
        det(R)                should be +1

    # Arguments
        R: shape (3, 3) or (4, 4)
    '''
    R = R[:3,:3]
    with np.printoptions(precision=6, suppress=True):
        print()
        print(R@R.T)
        print(np.linalg.norm(R, axis=-2))
        print(np.linalg.norm(R, axis=-1))
        print('%.6f'%np.linalg.det(R))

def transform_points(xyz, T):
    '''Transforms points by means of a homogeneous matrix.

    # Arguments
        xyz: array of shape (..., 3)
        T: array of shape (4, 4)
    '''
    dtype = xyz.dtype
    xyz = np.concatenate([xyz,np.ones_like(xyz[...,:1])], axis=-1)
    xyz = T @ xyz[...,None]
    return xyz[...,:3,0].astype(dtype)

def normalize_points(xyz):
    '''Transforms points into the range [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]

    # Arguments
        xyz: array of shape (..., 3)
    '''
    xyz -= 0.5 * (np.max(xyz, axis=0) + np.min(xyz, axis=0))
    xyz /= np.max(np.abs(xyz))
    return xyz


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


# legacy
#euler2rot2 = euler2rot
#rot2euler2 = rot2euler
