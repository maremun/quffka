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

from basics import get_D, batch_simplex_matvec, pad_data, radius, \
        simplex_matvec
from butterfly import butterfly_params
from butterfly_matvecs import batch_butterfly_matvec, butterfly_matvec


def dense_map_data(x, M, w, f):
    d1 = M.shape[1]
    d = x.shape[1]
    if d1 > d:
        x = pad_data(x, d1, d)

    Mx = np.dot(M, x.T)
    if f is not None:
        Mx = f(Mx)
    if w is not None:
        multiply = lambda z: np.einsum('i,ij->ij', w, z)
        Mx = multiply(Mx)

    return Mx

@jit(nopython=True)
def get_batch_mx(x, cos, sin, perm):
    b = batch_butterfly_matvec(x, cos, sin, perm)
    Mx = batch_simplex_matvec(b)
    return Mx


@jit(nopython=True)
def fast_batch_mapping_matvec(x, n, r=None, b_params=None, even=False):
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
            Mx[:d+1, :] = r[0] * get_batch_mx(x, cos[0], sin[0], perm[0])
            w[:] = sqrt(d) / r[0]
            if even:
                Mx[d+1:, :] = r[1] * get_batch_mx(x, cos[1], sin[1],
                                                  perm[1])[D-(d+1), :]
                w[d+1:] = sqrt(d) / r[1]
            else:
                Mx[d+1:, :] = -Mx[:D-(d+1), :]
        else:
            Mx = r[0] * get_batch_mx(x, cos[0], sin[0], perm[0])[:D, :]
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
        div = t*(d+1)
        Mx[div:, :] = -Mx[:D-div, :]
        w[div:] = w[:D-div]
    return Mx, w


@jit(nopython=True)
def fast_batch_approx_arccos1(x, div, n, r, b_params, gamma=None):
    d = x.shape[0]
    D = get_D(d, n)

    Mx, w = fast_batch_mapping_matvec(x, n, r, b_params, False)
    Mx = np.maximum(Mx, 0)
    for i in range(D):
        Mx[i, :] *= w[i]

    Mxx = np.empty((div, Mx.shape[0]))
    Mxx[:, :] = Mx[:, :div].T

    Mz = np.empty((Mx.shape[1] - div, Mx.shape[0]))
    Mz[:, :] = Mx[:, div:].T
    K = 2 * np.dot(Mxx, Mz.T) / D
    return K


@jit(nopython=True)
def fast_batch_approx_arccos0(x, div, n, r, b_params, gamma=None):
    d = x.shape[0]
    D = get_D(d, n)

    Mx, w = fast_batch_mapping_matvec(x, n, r, b_params, True)
    for i in range(Mx.shape[0]):
        for j in range(Mx.shape[1]):
            if Mx[i, j] > 0.:
                Mx[i, j] = 1.
            elif Mx[i, j] == 0.:
                Mx[i, j] = 0.5
            elif Mx[i, j] < 0.:
                Mx[i, j] = 0.
    for i in range(D):
        Mx[i, :] *= w[i]

    Mxx = np.empty((div, Mx.shape[0]))
    Mxx[:, :] = Mx[:, :div].T

    Mz = np.empty((Mx.shape[1] - div, Mx.shape[0]))
    Mz[:, :] = Mx[:, div:].T
    K = 2 * np.dot(Mxx, Mz.T) / D
    K += 0.5 * (1 - np.mean(np.power(w, 2)))
    return K


@jit(nopython=True)
def fast_batch_approx_rbf(x, div, n, r, b_params, gamma=None):
    '''
    Args
    x: data matrix (stacked X and Y, eg. test + train)
    div: X.shape[0]
    '''
    d = x.shape[0] # input dimension
    nobj = x.shape[1] # number of objects
    D = get_D(d, n) # output dimension
    if gamma is None:
        gamma = 1. / d
    gamma = 1. / d
    sigma = 1.0 / sqrt(2. * gamma)
    Mx, w = fast_batch_mapping_matvec(x, n, r, b_params, True)
    Mx /= sigma
    features = np.empty((2 * D, nobj))
    features[:D, :] = np.cos(Mx)
    features[D:, :] = np.sin(Mx)

    for i in range(D):
        features[i, :] *= w[i]
        features[D + i, :] *= w[i]

    MX = np.empty((div, features.shape[0]))
    MX[:, :] = features[:, :div].T

    MZ = np.empty((features.shape[1] - div, features.shape[0]))
    MZ[:, :] = features[:, div:].T
    K = np.dot(MX, MZ.T) / D
    K += 1. - np.mean(np.power(w, 2))
    return K


@jit(nopython=True)
def fast_batch_approx_angular(x, div, n, r, b_params, gamma=None):
    d = x.shape[0]
    D = get_D(d, n)

    Mx, w = fast_batch_mapping_matvec(x, n, r, b_params, True)
    Mx = np.sign(Mx)
    for i in range(D):
        Mx[i, :] *= w[i]

    Mxx = np.empty((div, Mx.shape[0]))
    Mxx[:, :] = Mx[:, :div].T

    Mz = np.empty((Mx.shape[1] - div, Mx.shape[0]))
    Mz[:, :] = Mx[:, div:].T
    K = np.dot(Mxx, Mz.T) / D
    return K


@jit(nopython=True)
def fast_batch_approx_linear(x, div, n, r, b_params):
    d = x.shape[0]
    D = get_D(d, n)
    Mx, w = fast_batch_mapping_matvec(x, n, r, b_params, False)
    for i in range(D):
        Mx[i, :] *= w[i]

    # TODO make arrays that are used in dot product C-contiguous
    Mxx = np.empty((div, Mx.shape[0]))
    Mxx[:, :] = Mx[:, :div].T

    Mz = np.empty((Mx.shape[1] - div, Mx.shape[0]))
    Mz[:, :] = Mx[:, div:].T
    K = np.dot(Mxx, Mz.T) / D
    return K


@jit(nopython=True)
def get_mx(x, cos, sin, p):
    b = butterfly_matvec(x, cos, sin, p)
    Mx = simplex_matvec(b)
    return Mx


@jit(nopython=True)
def fast_mapping_matvec(x, n, r, cos, sin, p):
    '''
              |V^T Q_0^T x|
    Mx = \rho |    ...    |
              |V^T Q_n^T x|
    Args:
    =====
    x: the data vector of size n.
    n: the parameter defining the new number of features.

    Returns:
    ========
    Mx: the mapping Mx.
    w: the weights.
    '''
    d = x.shape[0]
    D = get_D(d, n)
    Mx = np.empty((D,))
    if n < 1:
        if n >= 0.5:
            Mx[:d+1] = r[0] * get_mx(x, cos[0, :], sin[0, :], p[0, :])
            Mx[d+1:] = -Mx[:D-(d+1)]
        else:
            Mx = r[0] * get_mx(x, cos[0, :], sin[0, :], p[0, :])[:D]
        # TODO check out w
        w = np.ones(D) * sqrt(d) / r[0]
        return Mx, w

    n = int(np.ceil(n))
    w = np.empty(D)
    for i in range(n):
        w[i*(d+1):(i+1)*(d+1)] = sqrt(d) / r[i]
        Mx[i*(d+1):(i+1)*(d+1)] = \
                r[i] * get_mx(x, cos[i, :], sin[i, :], p[i, :])
    div = n*(d+1)
    Mx[div:] = -Mx[:D-div]
    w[div:] = w[:D-div]
    return Mx, w


@jit(nopython=True)
def fast_approx_arccos(x, z, n, r=None, b_params=None):
    d = x.shape[1]
    nobjx = x.shape[0]
    nobjz = z.shape[0]
    D = get_D(d, n)
    if n != int(np.ceil(n)):
        D -= 1
    Mx = np.empty((nobjx, D))
    Mz = np.empty((nobjz, D))

    Mx[0, :], w = fast_mapping_matvec(x[0, :], n, r, b_params)
    for i in range(1, nobjx):
        Mx[i, :] = fast_mapping_matvec(x[i, :], n, r, b_params)[0]

    Mz[0, :] = fast_mapping_matvec(z[0, :], n, r, b_params)[0]
    for i in range(1, nobjz):
        Mz[i, :] = fast_mapping_matvec(z[i, :], n, r, b_params)[0]

    Mx = np.maximum(Mx, 0)
    Mz = np.maximum(Mz, 0)

    for i in range(D):
        Mx[:, i] *= w[i]
        Mz[:, i] *= w[i]

    K = 2 * np.dot(Mx, Mz.T) / D
    return K
