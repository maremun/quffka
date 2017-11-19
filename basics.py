#!/usr/bin/env python3
#   encoding: utf-8
#   basics.py
# TODO documentation!
import numpy as np

from math import sqrt
from numba import jit


def pad_data(x, z, d1, n):
    pad = lambda xz: np.lib.pad(xz, ((0, 0), (0, d1-n)), 'constant', \
            constant_values=(0))
    x = pad(x)
    z = pad(z)
    return x, z


@jit(nopython=True)
def radius(n, k):
    h = int(np.ceil(k))
    rm = n + 2
    r = np.sqrt(2 * np.random.gamma(rm/2., 1., h))
    return r


@jit(nopython=True)
def get_d0(k, n):
    '''
    when using simplex method to generate points, we add also reflected ones,
    so the overall number of points created is 2 * (n+1).
    '''
    d0 = int(2 * k * (n+1))
    return d0


def generate_random_weights(k, n):
    d0 = get_d0(k, n)
    M = np.random.randn(d0, n)
    return M, None


@jit(nopython=True)
def simplex_matvec(x):
    '''
    V.T @ x (V is n-simplex)
    '''
    n = x.shape[0]
    mp = n + 1
    r = np.empty(n + 1)
    rv = np.empty(n)
    s = 0.
    for i in range(n):
        rv[i] = sqrt(mp / ((n - i) * n * (n - i + 1.))) * x[i]
        r[i] = s + rv[i] * (n-i)
        s += -rv[i]
    r[n] = s

    return r


@jit(nopython=True)
def batch_simplex_matvec(x):
    nobj = x.shape[1]
    n = x.shape[0]
    mp = n + 1
    r = np.empty((n+1, nobj))
    rv = np.empty(n)
    s = np.zeros(nobj)
    for i in range(n):
        rv[i] = sqrt(mp / ((n - i) * n * (n - i + 1.)))
        for o in range(nobj):
            rvo = rv[i] * x[i, o]
            r[i, o] = s[o] + rvo * (n-i)
            s[o] += -rvo
    for o in range(nobj):
        r[n, o] = s[o]
    return r


@jit(nopython=True)
def rnsimp(m):
    S = np.zeros((m, m + 1))
    mp = m + 1
    for i in range(m):
        rv = np.sqrt(mp / ((m - i) * m * (m - i + 1.)))
        S[i, i] = (m - i) * rv
        S[i, i+1:m+1] = -rv

    return S
