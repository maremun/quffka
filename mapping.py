#!/usr/bin/env python3
#   encoding: utf-8
#   mapping.py

# TODO No other method except quadrature-based does not use weights. Quadrature
# based is used via batch matvec function, so no point in using w here?

# TODO fast_batch_approx_kernel should have kwargs to be able to receive kernel
# hyperparameters
import numpy as np

from math import sqrt
from numba import jit

from basics import batch_simplex_matvec, get_D, radius
from butterfly import butterfly_params
from butterfly_matvecs import batch_butterfly_matvec


@jit(nopython=True)
def get_batch_mx(x, cos, sin, perm):
    b = batch_butterfly_matvec(x, cos, sin, perm)
    Mx = batch_simplex_matvec(b)
    return Mx


@jit(nopython=True)
def butt_quad_mapping(x, n, r=None, b_params=None, even=False):
    '''
              |V^T Q_0^T x|
    Mx = \rho |    ...    |
              |V^T Q_n^T x|
    Args:
    =====
    x: the data vector of dimension d.
    n: the parameter defining the new number of features.

    Returns:
    ========
    Mx: the mapping Mx.
    w: the weights.
    '''
    nobj = x.shape[1]
    d = x.shape[0]
    D = get_D(d, n)
    Mx = np.empty((D, nobj))
    if even:
        t = int(np.ceil(2*n))
    else:
        t = int(np.ceil(n))
    if r is None:
        r = radius(d, t)
    if b_params is None:
        b_params = butterfly_params(d, t)

    w = np.empty(D)
    cos, sin, perm = b_params
    #TODO fix this case when generating for rbf-like kernels
    if n < 1:
        if n > 0.5:
# TODO check indexing! d+1?
            Mx[:d+1, :] = r[0] * get_batch_mx(x, cos[0], sin[0], perm[0])
            w[:] = sqrt(d) / r[0]
            if even:
                Mx[d+1:, :] = r[1] * get_batch_mx(x, cos[1], sin[1],
                                                  perm[1])[D-(d+1), :]
                w[d+1:] = sqrt(d) / r[1]
            else:
                Mx[d+1:, :] = -Mx[:D-(d+1), :]
        else:
            Mx[:, :] = r[0] * get_batch_mx(x, cos[0], sin[0], perm[0])[:D, :]
            w[:] = sqrt(d) / r[0]
        return Mx, w

    if even:
        for i in range(t-1):
            w[i*(d+1):(i+1)*(d+1)] = sqrt(d) / r[i]
            Mx[i*(d+1):(i+1)*(d+1), :] = \
                    r[i] * get_batch_mx(x, cos[i], sin[i], perm[i])
        i = t - 1
        Mx[i*(d+1):, :] = r[i] * get_batch_mx(x, cos[i], sin[i],
                                              perm[i])
        w[i*(d+1):] = sqrt(d) / r[i]

    else:
        for i in range(t):
            w[i*(d+1):(i+1)*(d+1)] = sqrt(d) / r[i]
            Mx[i*(d+1):(i+1)*(d+1), :] = \
                    r[i] * get_batch_mx(x, cos[i], sin[i], perm[i])
        div = t * (d+1)
        Mx[div:, :] = -Mx[:D-div, :]
        w[div:] = w[:D-div]

    return Mx, w
