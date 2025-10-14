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

    # cutoff for small singular values to avoid numerical issuses, division by zero etc.
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


