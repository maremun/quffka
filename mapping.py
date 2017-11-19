#!/usr/bin/env python3
#   encoding: utf-8
#   mapping.py

import numpy as np
from math import sqrt
from numba import jit
from butterfly import batch_butterfly_matvec, butterfly_matvec
from utils import get_d0, batch_simplex_matvec, simplex_matvec, \
                  butterfly_params, radius


@jit(nopython=True)
def get_batch_mx(x, cos, sin, perm):
    b = batch_butterfly_matvec(x, cos, sin, perm)
    Mx = batch_simplex_matvec(b)
    return Mx


@jit(nopython=True)
def fast_batch_mapping_matvec(x, k, r=None, b_params=None, even=False):
    '''
              |V^T Q_0^T x|
    Mx = \rho |    ...    |
              |V^T Q_k^T x|
    Args:
    =====
    x: the data vector of size n.
    k: the parameter defining the new number of features.

    Returns:
    ========
    Mx: the mapping Mx.
    w: the weights.
    '''
    nobj = x.shape[1]
    n = x.shape[0]
    d0 = get_d0(k, n)
    Mx = np.empty((d0, nobj))
    if even:
        t = int(np.ceil(2*k))
    else:
        t = int(np.ceil(k))
    if r is None:
        r = radius(n, t)
    if b_params is None:
        b_params = butterfly_params(n, t)

    w = np.empty(d0)
    cos, sin, perm = b_params
    #TODO fix this case when generating for rbf-like kernels
    if k < 1:
        if k > 0.5:
            Mx[:n+1, :] = r[0] * get_batch_mx(x, cos[0], sin[0], perm[0])
            w[:] = sqrt(n) / r[0]
            if even:
                Mx[n+1:, :] = r[1] * get_batch_mx(x, cos[1], sin[1],
                                                  perm[1])[d0-(n+1), :]
                w[n+1:] = sqrt(n) / r[1]
            else:
                Mx[n+1:, :] = -Mx[:d0-(n+1), :]
        else:
            Mx = r[0] * get_batch_mx(x, cos[0], sin[0], perm[0])[:d0, :]
            w[:] = sqrt(n) / r[0]
        return Mx, w

    if even:
        for i in range(t-1):
            w[i*(n+1):(i+1)*(n+1)] = sqrt(n) / r[i]
            Mx[i*(n+1):(i+1)*(n+1), :] = \
                    r[i] * get_batch_mx(x, cos[i], sin[i], perm[i])
            i = t - 1
            Mx[i*(n+1):, :] = r[i] * get_batch_mx(x, cos[i], sin[i],
                                                  perm[i])
            w[i*(n+1):] = sqrt(n) / r[i]

    else:
        for i in range(t):
            w[i*(n+1):(i+1)*(n+1)] = sqrt(n) / r[i]
            Mx[i*(n+1):(i+1)*(n+1), :] = \
                    r[i] * get_batch_mx(x, cos[i], sin[i], perm[i])
        div = t*(n+1)
        Mx[div:, :] = -Mx[:d0-div, :]
        w[div:] = w[:d0-div]

    return Mx, w


@jit(nopython=True)
def fast_batch_approx_arccos1(x, div, k, r, b_params, gamma=None):
    #numba does not support kwargs or args properly
    n = x.shape[0]
    nsamples = get_d0(k, n)

    Mx, w = fast_batch_mapping_matvec(x, k, r, b_params, False)
    Mx = np.maximum(Mx, 0)
    for i in range(nsamples):
        Mx[i, :] *= w[i]

    Mxx = np.empty((div, Mx.shape[0]))
    Mxx[:, :] = Mx[:, :div].T

    Mz = np.empty((Mx.shape[1] - div, Mx.shape[0]))
    Mz[:, :] = Mx[:, div:].T
    K = 2 * np.dot(Mxx, Mz.T) / nsamples

    return K


@jit(nopython=True)
def fast_batch_approx_arccos0(x, div, k, r, b_params, gamma=None):
    #numba does not support kwargs or args properly
    n = x.shape[0]
    nsamples = get_d0(k, n)

    Mx, w = fast_batch_mapping_matvec(x, k, r, b_params, True)
    for i in range(Mx.shape[0]):
        for j in range(Mx.shape[1]):
            if Mx[i, j] > 0.:
                Mx[i, j] = 1.
            elif Mx[i, j] == 0.:
                Mx[i, j] = 0.5
            elif Mx[i, j] < 0.:
                Mx[i, j] = 0.
    for i in range(nsamples):
        Mx[i, :] *= w[i]

    Mxx = np.empty((div, Mx.shape[0]))
    Mxx[:, :] = Mx[:, :div].T

    Mz = np.empty((Mx.shape[1] - div, Mx.shape[0]))
    Mz[:, :] = Mx[:, div:].T
    K = 2 * np.dot(Mxx, Mz.T) / nsamples
    K += 0.5 * (1 - np.mean(np.power(w, 2)))

    return K


@jit(nopython=True)
def fast_batch_approx_rbf(x, div, k, r, b_params, gamma=None):
    '''
    Args
    x: data matrix (stacked X and Y, eg. test + train)
    div: X.shape[0]
    '''
    n = x.shape[0] # input dimension
    nobj = x.shape[1] # number of objects
    nsamples = get_d0(k, n) # output dimension
    if gamma is None:
        gamma = 1. / n
    gamma = 1. / n
    sigma = 1.0 / sqrt(2 * gamma)
    Mx, w = fast_batch_mapping_matvec(x, k, r, b_params, True)
    Mx /= sigma
    features = np.empty((2 * nsamples, nobj))
    features[:nsamples, :] = np.cos(Mx)
    features[nsamples:, :] = np.sin(Mx)

    for i in range(nsamples):
        features[i, :] *= w[i]
        features[nsamples + i, :] *= w[i]

    MX = np.empty((div, features.shape[0]))
    MX[:, :] = features[:, :div].T

    MZ = np.empty((features.shape[1] - div, features.shape[0]))
    MZ[:, :] = features[:, div:].T
    K = np.dot(MX, MZ.T) / nsamples

    K += 1 - np.mean(np.power(w, 2))

    return K


@jit(nopython=True)
def fast_batch_approx_angular(x, div, k, r, b_params, gamma=None):
    #numba does not support kwargs or args properly
    n = x.shape[0]
    nsamples = get_d0(k, n)

    Mx, w = fast_batch_mapping_matvec(x, k, r, b_params, True)
    Mx = np.sign(Mx)
    for i in range(nsamples):
        Mx[i, :] *= w[i]

    Mxx = np.empty((div, Mx.shape[0]))
    Mxx[:, :] = Mx[:, :div].T

    Mz = np.empty((Mx.shape[1] - div, Mx.shape[0]))
    Mz[:, :] = Mx[:, div:].T
    K = np.dot(Mxx, Mz.T) / nsamples

    return K


@jit(nopython=True)
def fast_batch_approx_linear(x, div, k, r, b_params):
    n = x.shape[0]
    nsamples = get_d0(k, n)
    Mx, w = fast_batch_mapping_matvec(x, k, r, b_params, False)
    for i in range(nsamples):
        Mx[i, :] *= w[i]

    # TODO make arrays that are used in dot product C-contiguous
    Mxx = np.empty((div, Mx.shape[0]))
    Mxx[:, :] = Mx[:, :div].T

    Mz = np.empty((Mx.shape[1] - div, Mx.shape[0]))
    Mz[:, :] = Mx[:, div:].T
    K = np.dot(Mxx, Mz.T) / nsamples

    #Mz = Mx[:, div:]
    #Mx = Mx[:, :div]
    #K = np.dot(Mx.T, Mz) / nsamples

    return K


@jit(nopython=True)
def get_mx(x, cos, sin, p):
    b = butterfly_matvec(x, cos, sin, p)
    Mx = simplex_matvec(b)
    return Mx


@jit(nopython=True)
def fast_mapping_matvec(x, k, r, cos, sin, p):
    '''
              |V^T Q_0^T x|
    Mx = \rho |    ...    |
              |V^T Q_k^T x|
    Args:
    =====
    x: the data vector of size n.
    k: the parameter defining the new number of features.

    Returns:
    ========
    Mx: the mapping Mx.
    w: the weights.
    '''
    n = x.shape[0]
    d0 = get_d0(k, n)
    Mx = np.empty((d0, ))
    if k < 1:
        if k >= 0.5:
            Mx[:n+1] = r[0] * get_mx(x, cos[0, :], sin[0, :], p[0, :])
            Mx[n+1:] = -Mx[:d0-(n+1)]
        else:
            Mx = r[0] * get_mx(x, cos[0, :], sin[0, :], p[0, :])[:d0]
        # TODO check out w
        w = np.ones(d0) * sqrt(n) / r[0]
        return Mx, w

    k = int(np.ceil(k))
    w = np.empty(d0)
    for i in range(k):
        w[i*(n+1):(i+1)*(n+1)] = sqrt(n) / r[i]
        Mx[i*(n+1):(i+1)*(n+1)] = \
                r[i] * get_mx(x, cos[i, :], sin[i, :], p[i, :])
    div = k*(n+1)
    Mx[div:] = -Mx[:d0-div]
    w[div:] = w[:d0-div]

    return Mx, w


@jit(nopython=True)
def fast_approx_arccos(x, z, k, r=None, b_params=None):
    n = x.shape[1]
    nobjx = x.shape[0]
    nobjz = z.shape[0]
    nsamples = get_d0(k, n)
    if k != int(np.ceil(k)):
        nsamples -= 1
    Mx = np.empty((nobjx, nsamples))
    Mz = np.empty((nobjz, nsamples))

    Mx[0, :], w = fast_mapping_matvec(x[0, :], k, r, b_params)
    for i in range(1, nobjx):
        Mx[i, :] = fast_mapping_matvec(x[i, :], k, r, b_params)[0]

    Mz[0, :] = fast_mapping_matvec(z[0, :], k, r, b_params)[0]
    for i in range(1, nobjz):
        Mz[i, :] = fast_mapping_matvec(z[i, :], k, r, b_params)[0]

    Mx = np.maximum(Mx, 0)
    Mz = np.maximum(Mz, 0)

    for i in range(nsamples):
        Mx[:, i] *= w[i]
        Mz[:, i] *= w[i]

    K = 2 * np.dot(Mx, Mz.T) / nsamples

    return K
