"""
SPDX-License-Identifier: MIT
Copyright © 2015 - 2025 Markus Völk
Code was taken from https://github.com/mvoelk/utils
"""

import numpy as np


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



def dlsinv(A, d=0, rcond=1e-15):
    """Damped Least Squares Inverse (DLS)

    # Arguments
        A : array_like, shape (M, N)
            Matrix to be inverted
        d : float, optional
            Damping factor
        rcond : float
            Cutoff for small singular values.
            Singular values smaller than
            `rcond` * largest_singular_value
            are set to zero.

    # Returns
        B : array, shape (N, M)
            DLS inverse of A
    """

    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    m = U.shape[0]
    n = Vt.shape[1]
    r = min(n, m)
    cutoff = rcond*np.max(s)

    if d == 0:
        # least squares
        for i in range(r):
            if s[i] < cutoff:
                s[i] = 0.0
            else:
                s[i] = 1./s[i]
    else:
        # damped least squares
        d = d*d
        for i in range(r):
            if s[i] < cutoff:
                s[i] = 0.0
            else:
                s[i] = s[i]/(s[i]*s[i]+d)

    B = np.dot(Vt.T, np.multiply(s[:, np.newaxis], U.T))
    return B


def srinv(A, d=0.1, w_0=1.0, rcond=1e-15):
    """Singularity Robust Inverse (SR-Inverse) according to [1]

    [1] Y. Nakamura and H. Hanafusa, "Inverse Kinematic Solutions With
        Singularity Robustness for Robot Manipulator Control,"
        J. Dyn. Syst. Meas. Control, vol. 108, no. 3, p. 163, Sep. 1986.

    # Arguments
        A : array_like, shape (M, N)
            Matrix to be inverted
        d : float, optional
            Damping factor
        w_0 : float, optional
            Manipulability threshold
        rcond : float
            Cutoff for small singular values.
            Singular values smaller than
            `rcond` * largest_singular_value
            are set to zero.

    # Returns
        B : array, shape (N, M)
            SR-Inverse of A
    """

    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    m = U.shape[0]
    n = Vt.shape[1]
    r = min(n, m)
    cutoff = rcond*np.max(s)

    # singulartity robust
    d = d*d
    w = np.sqrt(np.linalg.det(np.dot(A,A.T)))
    for i in range(r):
        if s[i] < cutoff:
            s[i] = 0.0
        else:
            if w < w_0:
                s[i] = s[i]/(s[i]*s[i]+d*(1-w/w_0))

    B = np.dot(Vt.T, np.multiply(s[:, np.newaxis], U.T))
    return B


