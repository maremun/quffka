#   encoding: utf-8
#   mapping.py

# TODO No other method except quadrature-based does not use weights. Quadrature
# based is used via batch matvec function, so no point in using w here?

# TODO fast_batch_approx_kernel should have kwargs to be able to receive kernel
# hyperparameters
import numpy as np

from math import sqrt
from numba import jit

from .basics import batch_simplex_matvec, get_D, radius
from .butterfly import butterfly_params
from .butterfly_matvecs import batch_butterfly_matvec


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
    if n < 1:
        if n > 0.5:
# TODO check indexing! d+1?
            d0 = int(get_D(d, 1)/2)
            rr = np.ones((d0, 1))
            rr[:, 0] = r[:d0]
            Mx[:d+1, :] = rr * get_batch_mx(x, cos[0], sin[0], perm[0])
            w[:] = sqrt(d) / rr[:, 0]
            if even:
                dd = d0
                d0 = r.shape[0]
                rr = np.ones((d0-dd, 1))
                rr[:, 0] = r[d0:]
                Mx[d+1:, :] = rr * get_batch_mx(x, cos[1], sin[1],
                                                perm[1])[D-(d+1), :]
                w[d+1:] = sqrt(d) / rr[:, 0]
            else:
                Mx[d+1:, :] = -Mx[:D-(d+1), :]
        else:
            Mx[:, :] = r[:] * get_batch_mx(x, cos[0], sin[0], perm[0])[:D, :]
            w[:] = sqrt(d) / r[:]
        return Mx, w

    if even:
        dd = 0
        for i in range(t-1):
            d0 = int(get_D(d, i+1) / 2)
            rr = np.ones((d0-dd, 1))
            rr[:, 0] = r[dd:d0]
            w[i*(d+1):(i+1)*(d+1)] = sqrt(d) / rr[:, 0]
            Mx[i*(d+1):(i+1)*(d+1), :] = \
                    rr * get_batch_mx(x, cos[i], sin[i], perm[i])
            dd = d0

        i = t - 1
        d0 = int(get_D(d, i+1) / 2)
        rr = np.ones((d0-dd, 1))
        rr[:, 0] = r[dd:d0]
        Mx[i*(d+1):, :] = rr * get_batch_mx(x, cos[i], sin[i], perm[i])
        w[i*(d+1):] = sqrt(d) / rr[:, 0]

    else:
        dd = 0
        for i in range(t):
            d0 = int(get_D(d,(i+1)) / 2)
            rr = np.ones((d0-dd, 1))
            rr[:, 0] = r[dd:d0]
            w[i*(d+1):(i+1)*(d+1)] = sqrt(d) / rr[:, 0]
            Mx[i*(d+1):(i+1)*(d+1), :] = \
                    rr * get_batch_mx(x, cos[i], sin[i], perm[i])
            dd = d0
        div = t * (d+1)
        Mx[div:, :] = -Mx[:D-div, :]
        w[div:] = w[:D-div]

    return Mx, w
