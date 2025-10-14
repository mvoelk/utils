"""
SPDX-License-Identifier: MIT
Copyright © 2015 - 2025 Markus Völk
Code was taken from https://github.com/mvoelk/utils
"""

import sympy as sp


def skew(v):
    x, y, z = v
    return sp.Matrix([
        [0, -z,  y],
        [z,  0, -x],
        [-y, x,  0]
    ])

def unskew(S):
    return sp.Matrix([S[2, 1], S[0, 2], S[1, 0]])


def rotx(angle):
    s, c = sp.sin(angle), sp.cos(angle)
    return sp.Matrix([[1,0,0], [0,c,-s], [0,s,c]])

def roty(angle):
    s, c = sp.sin(angle), sp.cos(angle)
    return sp.Matrix([[c,0,s], [0,1,0], [-s,0,c]])

def rotz(angle):
    s, c = sp.sin(angle), sp.cos(angle)
    return sp.Matrix([[c,-s,0], [s,c,0], [0,0,1]])


def trafo(t=sp.Matrix([0,0,0]), R=sp.eye(3)):
    '''Builds a homogeneous transformation matrix (symbolic version)

    # Arguments
        t: Matrix (3)
        R: Matrix (3,3), rotation matrix

    # Returns
        T: Matrix (4,4) homogeneous transformation matrix
    '''
    T = sp.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T

def hinv(T):
    '''Inverts a homogeneous transformation matrix (symbolic version)

    # Arguments
        T: Matrix (4,4)

    # Return
        Ti: Matrix (4,4), the inverse of T
    '''
    Rt = T[:3,:3].T
    t = T[:3,3]
    Ti = sp.eye(4)
    Ti[:3,:3] = Rt
    Ti[:3,3] = -Rt * t
    return Ti


def euler2rot(angles, axes='xyz', fixed=False):
    '''Converts arbitrary Euler Angles to Rotation Matrix (symbolic version)

    # Arguments
        angles: list or tuple of symbolic variables or expressions
        axes: string like 'xyz', 'zyx', etc.

            V-Rep: 'xyz' False
            Roll-Pitch-Yaw, ROS: 'xyz' True
            Yaw-Pitch-Roll: 'zyx' False

        fixed: if True, uses fixed frame (extrinsic), else rotating (intrinsic)

    # Return
        R: Matrix (3,3)

    '''
    Rs = [[rotx, roty, rotz]['xyz'.index(c)](a) for a, c in zip(angles, axes.lower())]
    if fixed:
        Rs = Rs[::-1]
    R = Rs[0]
    for Ri in Rs[1:]:
        R = R * Ri
    return R

def rot2euler(R, axes='xyz', fixed=False, s=1):
    '''Converts a Rotation Matrix to Euler Angles (symbolic version)

    # Arguments
        R: Matrix (3,3) rotation matrix
        axes: string, e.g. 'xyz'
        fixed: boolean
        s: sign multiplier (1 or -1)

    # Return
        Matrix([angle1, angle2, angle3]) or None if axes is not implemented
    '''
    if axes == 'xyz':
        if fixed:
            sy = sp.sqrt(sp.Max(R[0,0]**2 + R[1,0]**2, 0))
            return sp.Matrix([
                sp.atan2(R[2,1], R[2,2]),
                sp.atan2(-R[2,0], sy),
                sp.atan2(R[1,0], R[0,0])
            ])
        else:
            val = sp.sqrt(sp.Max(1 - R[0,2]**2, 0))
            return sp.Matrix([
                sp.atan2(-s * R[1,2], s * R[2,2]),
                sp.atan2(R[0,2], s * val),
                sp.atan2(-s * R[0,1], s * R[0,0])
            ])
    else:
        return None


def rot2vec(R):
    arg = 0.5 * (R[0,0] + R[1,1] + R[2,2] - 1)
    #arg = sp.Min(1, sp.Max(-1, arg))  # symbolic clip
    a = sp.acos(arg)
    
    # Small-angle and pi-angle conditions can’t be evaluated symbolically,
    # so we define v symbolically for the general case.
    v = sp.Matrix([
        R[2,1] - R[1,2],
        R[0,2] - R[2,0],
        R[1,0] - R[0,1]
    ])
    r = v / v.norm() * a
    return r

def vec2rot(r):
    a = sp.sqrt(r.dot(r))
    v = r / a
    x, y, z = v
    s, c = sp.sin(a), sp.cos(a)
    t = 1 - c
    xs, ys, zs = x*s, y*s, z*s
    xyt, xzt, yzt = x*y*t, x*z*t, y*z*t
    R = sp.Matrix([
        [c+x**2*t, xyt-zs, xzt+ys],
        [xyt+zs, c+y**2*t, yzt-xs],
        [xzt-ys, yzt+xs, c+z**2*t],
    ])
    return sp.simplify(R)



def replace_small(expr, tol=1e-10):
    def repl(x):
        if x.is_Number:
            if abs(float(x.evalf())) < tol:
                return sp.S(0)
        return x
    return expr.xreplace({n: repl(n) for n in expr.atoms(sp.Number)})


rotxs, rotys, rotzs, trafos, hinvs = rotx, roty, rotz, trafo, hinv
rot2eulers, euler2rots, rot2vecs, vec2rots  = rot2euler, euler2rot, rot2vec, vec2rot
