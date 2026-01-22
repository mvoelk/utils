"""
SPDX-License-Identifier: MIT
Copyright © 2015 - 2025 Markus Völk
Code was taken from https://github.com/mvoelk/utils
"""

import math
import numpy as np


def skew(v):
    x, y, z = v
    S = np.array([[ 0, -z,  y],
                  [ z,  0, -x],
                  [-y,  x,  0]], dtype=float)
    return S

def unskew(S):
    v = np.array([S[2,1], S[0,2], S[1,0]])
    return v


def dh2trafo(d, r, alpha, theta, modified=False):
    """Creates a homogeneous matrix from given Denavit Hartenberg parameters"""
    
    sα, cα = np.sin(alpha), np.cos(alpha)
    sθ, cθ = np.sin(theta), np.cos(theta)

    if modified:
        T = np.array([
            [cθ,    -sθ,   0,   r    ],
            [sθ*cα, cθ*cα, -sα, -d*sα],
            [sθ*sα, cθ*sα,  cα,  d*cα],
            [0,     0,     0,   1    ],
        ])
    else:
        T = np.array([
            [cθ, -sθ*cα, sθ*sα, r*cθ],
            [sθ, cθ*cα, -cθ*sα, r*sθ],
            [0,  sα,     cα,    d   ],
            [0,  0,      0,     1   ],
        ])

    return T


def dlsinv(J, d=0.1, w_0=0.01):
    """Damped Least Squares Inverse (DLS)

    # Arguments
        J: array, shape (m, n), Matrix to be inverted
            for the typical robot case m ≤ n
        d: float, Damping factor (Nakamura, 1986)
            if 0 or None, least squares solution is calculated
        w_0: float, Manipulability threshold (Yoshikawa, 1985)
            if 0 or None, damped least squares solution is calculated
            otherwise an adaptive damping strategy is used

    # Return
        array, shape (n, m), DLS inverse of J
    """

    # cutoff for small singular values to avoid numerical issues, division by zero etc.
    # singular values smaller then rcond * largest_singular_value are set to zero
    rcond = 1e-12
    
    m, n = J.shape
    #r = min(m,n) # rank of J

    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    
    r = sum(s > rcond*np.max(s))

    s[r:] = 0.0

    if not d:
        # least squares
        s[:r] = 1. / s[:r]
    elif not w_0:
        # damped least squares
        s[:r] = s[:r] / (s[:r]*s[:r] + d*d)
    else:
        # adaptive damping
        w = np.prod(s) # manipulability ω, same as sqrt(det(J@J.T))
        if w < w_0:
            # damping near singularities
            s[:r] = s[:r] / (s[:r]*s[:r] + d*d*(1-w/w_0))
        else:
            # no damping far from singularies
            s[:r] = 1. / s[:r]

    Ji = Vt.T @ (s[:,None] * U.T)
    
    return Ji



def rot_to_euler(R, axes='xyz', fixed=False):

    match axes:
        case 'zyx':
            if fixed:
                raise NotImplementedError
            else:
                return np.array([
                        math.atan2(R[1,0], R[0,0]),
                        math.atan2(-R[2,0], (R[2,1]**2+R[2,2]**2)**0.5),
                        math.atan2(R[2,1], R[2,2]),
                    ])
        case 'xyz':
            if fixed:
                raise NotImplementedError
            else:
                return np.array([
                        math.atan2(-R[1,2], R[2,2]),
                        math.atan2(R[0,2], (R[0,0]**2+R[0,1]**2)**0.5),
                        math.atan2(-R[0,1], R[0,0]),
                    ])
        case 'zyz':
            if fixed:
                raise NotImplementedError
            else:
                return np.array([
                        math.atan2(R[1,2], R[0,2]),
                        math.atan2((R[0,2]**2+R[1,2]**2)**0.5, R[2,2]),
                        math.atan2(R[2,1], -R[2,0]),
                    ])
        case 'zxz':
            if fixed:
                raise NotImplementedError
            else:
                return np.array([
                        math.atan2(R[0,2], -R[1,2]),
                        math.atan2((R[0,2]**2+R[1,2]**2)**0.5, R[2,2]),
                        math.atan2(R[2,0], R[2,1]),
                    ])
        case _:
            raise NotImplementedError

def euler_to_rot(angles, axes='xyz', fixed=False):

    s1, s2, s3 = np.sin(angles)
    c1, c2, c3 = np.cos(angles)

    match axes:
        case 'zyx':
            if fixed:
                raise NotImplementedError
            else:
                return np.array([
                        [ c2*c1, c1*s3*s2-c3*s1, s3*s1+c3*c1*s2 ],
                        [ c2*s1, c3*c1+s3*s2*s1, c3*s2*s1-c1*s3 ],
                        [ -s2,   c2*s3,          c3*c2          ],
                    ])
        case 'xyz':
            if fixed:
                raise NotImplementedError
            else:
                return np.array([
                        [ c2*c3,          -c2*s3,         s2     ],
                        [ c1*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1 ],
                        [ s1*s3-c1*c3*s2, c3*s1+c1*s2*s3, c1*c2  ],
                    ])
        case 'zyz':
            if fixed:
                raise NotImplementedError
            else:
                return np.array([
                        [ c2*c1*c3-s1*s3, -c3*s1-c2*c1*s3, c1*s2 ],
                        [ c1*s3+c2*c3*s1, c1*c3-c2*s1*s3,  s2*s1 ],
                        [ -c3*s2,         s2*s3,           c2    ],
                    ])
        case 'zxz':
            if fixed:
                raise NotImplementedError
            else:
                return np.array([
                        [ c1*c3-c2*s1*s3, -c1*s3-c2*c3*s1, s2*s1  ],
                        [ c3*s1+c2*c1*s3, c2*c1*c3-s1*s3,  -c1*s2 ],
                        [ s2*s3,          c3*s2,           c2     ],
                    ])
        case _:
            raise NotImplementedError

def euler_rate_velocity_trafo(angles, axes='xyz', inv=False):
    """
    Calculates a transformation matrix from XYZ euler rates to angular velocities

    ROS ZYX (yaw-pitch-roll) fixed-axis (extrinsic)    
    """
    
    sa, sb, sc = np.sin(angles)
    ca, cb, cc = np.cos(angles)

    match axes:
        case 'zyx': # fixed = False
            if inv:
                return np.array([
                    [ca*sb/cb, sb*sa/cb, 1],
                    [-sa, ca, 0],
                    [ca/cb, sa/cb, 0],
                ])
            else:
                return np.array([
                    [0, -sa, cb*ca],
                    [0, ca, cb*sa],
                    [1, 0, -sb],
                ])
        case 'xyz': # fixed = False
            if inv:
                return np.array([
                    [1, sa*sb/cb, -ca*sb/cb],
                    [0, ca, sa],
                    [0, -sa/cb, ca/cb],
                ])
            else:
                return np.array([
                    [1, 0, sb],
                    [0, ca, -cb*sa],
                    [0, sa, ca*cb],
                ])
        case 'zyz': # fixed = False
            if inv:
                return np.array([
                    [-cb*ca/sb, -cb*sa/sb, 1],
                    [-sa, ca, 0],
                    [ca/sb, sa/sb, 0],
                ])
            else:
                return np.array([
                    [0, -sa, ca*sb],
                    [0, ca, sa*sb],
                    [1, 0, cb],
                ])
        case 'zxz': # fixed = False
            if inv:
                return np.array([
                    [-cb*sa/sb, cb*ca/sb, 1],
                    [ca, sa, 0],
                    [sa/sb, -ca/sb, 0],
                ])
            else:
                return np.array([
                    [0, ca, sa*sb],
                    [0, sa, -ca*sb],
                    [1, 0, cb],
                ])
        case _:
            raise NotImplementedError


def rot_to_vec(R):
    eps = 1e-6
    arg = 0.5 * (R[0,0]+R[1,1]+R[2,2] - 1)
    arg = np.clip(arg, -1, 1)
    a = np.arccos(arg)
    if a < eps:
        v = np.array([0,0,0])
    elif np.pi-a < eps:
        # for angle = pi, axis is the eigenvector corresponding to eigenvalue 1
        v = np.linalg.svd(R - np.eye(3))[0][:,-1]
    else:
        v = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
        v = v / np.linalg.norm(v)
        #v = v / (2 * np.sin(a))
    r = v * a
    return np.array(r, dtype=R.dtype)

def vec_to_rot(r):
    a = np.linalg.norm(r)
    v = r/a
    x, y, z = v
    s, c = np.sin(a), np.cos(a)
    t = 1.0 - c
    xs, ys, zs = x*s, y*s, z*s
    xyt, xzt, yzt = x*y*t, x*z*t, y*z*t
    R = np.array([
        [c+x**2*t, xyt-zs, xzt+ys],
        [xyt+zs, c+y**2*t, yzt-xs],
        [xzt-ys, yzt+xs, c+z**2*t],
    ])
    return R

def vec_rate_velocity_trafo(r, inv=False):
    a = np.linalg.norm(r)
    S, I = skew(r), np.eye(3)
    sa, ca = np.sin(a), np.cos(a)
    if inv:
        return I - 0.5*S + (1/a**2)*(1-(a*sa/(2*(1-ca))))*S@S
    else:
        return I + ((1-ca)/a**2)*S + ((a-sa)/a**3)*S@S



