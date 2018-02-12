#!/usr/bin/env python3
#   encoding: utf-8
#   basics.py
# TODO documentation!
import numpy as np

from math import sqrt
from numba import jit


def pad_data(x, d1, d):
    pad = lambda z: np.lib.pad(z, ((0, 0), (0, d1-d)),
                               'constant', constant_values=(0))
    x = pad(x)
    return x


@jit(nopython=True)
def radius(d, n):
    rm = d + 2
    D = int(get_D(d, n) / 2)
    r = np.sqrt(2 * np.random.gamma(rm/2., 1., D))
    return r


@jit(nopython=True)
def get_D(d, n):
    '''
    when using simplex method to generate points, we add also reflected ones,
    so the overall number of points created is 2 * (n+1) + 1, where 1 comes
    from the weight of function at zero.
    '''
    D = int(n * (2 * (d+1)))  # + 1))
    return D


def generate_random_weights(d, n):
    '''
    Generates random Fourier features

    Args
        d: the input dimension of data
        n: total number of features is D = 2*n(d+1)

    Returns:
        M: the points to estimate integral with
        w: the weights of the points in M
    '''
    D = get_D(d, n)
    M = np.random.randn(D, d)
    w = None  # all points have equal weights
    return M, w


@jit(nopython=True)
def rnsimp(m):
    S = np.zeros((m, m + 1))
    mp = m + 1
    for i in range(m):
        rv = np.sqrt(mp / ((m - i) * m * (m - i + 1.)))
        S[i, i] = (m - i) * rv
        S[i, i+1:m+1] = -rv

    return S


@jit(nopython=True)
def simplex_matvec(x):
    '''
    V.T @ x (V is n-simplex)
    '''
    d = x.shape[0]
    mp = d + 1
    r = np.empty(d + 1)
    rv = np.empty(d)
    s = 0.
    for i in range(d):
        rv[i] = sqrt(mp / ((d-i) * d * (d-i+1.))) * x[i]
        r[i] = s + rv[i] * (d-i)
        s += -rv[i]
    r[d] = s

    return r


@jit(nopython=True)
def batch_simplex_matvec(x):
    nobj = x.shape[1]
    d = x.shape[0]
    mp = d + 1
    r = np.empty((d+1, nobj))
    rv = np.empty(d)
    s = np.zeros(nobj)
    for i in range(d):
        rv[i] = sqrt(mp / ((d-i) * d * (d-i+1.)))
        for o in range(nobj):
            rvo = rv[i] * x[i, o]
            r[i, o] = s[o] + rvo * (d-i)
            s[o] += -rvo
    for o in range(nobj):
        r[d, o] = s[o]
    return r
