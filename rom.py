#!/usr/bin/env python3
#   encoding: utf-8
#   rom.py


import numpy as np
from utils import get_d0, rnsimp, radius, butterfly_params
from butterfly import butterfly
#from scipy.stats import chi
from numba import jit
from math import sqrt

@jit(nopython=True)
def hadamard(n):
    if n < 1:
        lg2 = 0
    else:
        lg2 = int(np.log2(n))
    if 2 ** lg2 != n:
        raise ValueError("n must be an positive integer, and n must be "
                         "a power of 2")

    H = np.zeros((n, n))
    H[0, 0] = 1

    # Sylvester's construction
    for i in range(0, lg2):
        p = 2**i
        H[:p, p:2*p] = H[:p, :p]
        H[p:2*p, :p] = H[:p, :p]
        H[p:2*p, p:2*p] = -H[:p, :p]

    H /= sqrt(n)
    return H


@jit(nopython=True)
def diagonal(n):
    '''
    Generates diagonal matrix D if size n x n with iid Rademacher random
        variables on the diagonal.
    '''
    d = (np.random.randint(0, 2, size=n) - 0.5) * 2
    D = np.diag(d)

    return D


@jit(nopython=True)
def single(p, n):
    S = hadamard(n)
    D = diagonal(n)
    M = np.dot(S, D)
    for _ in range(p-1):
        D = diagonal(n)
        M = np.dot(M, np.dot(S, D))
    return M


@jit(nopython=True)
def generate_rademacher_weights(k, n, p=3):
    '''
    Generates n x n S-Rademacher random matrix as in
        https://arxiv.org/abs/1703.00864.
    '''
    c = np.log2(n)
    f = np.floor(c)
    d0 = get_d0(k, n)
    if f != c:
        n = int(2 ** (f + 1))

    M = np.zeros((d0, n))

    t = int(np.ceil(d0/n))
    for i in range(t-1):
        M[i*n:(i+1)*n, :] = single(p, n)
    i = t-1
    M[i*n:, :] = single(p, n)[:d0 - i*n, :]
    M = np.sqrt(n) * M[:d0, :]

    return M, None


@jit(nopython=True)
def generate_gort_weights(k, n):
    d0 = get_d0(k, n)
    if d0 < n:
        G = np.random.randn(n, d0)
        Q, _ = np.linalg.qr(G)
        Q = Q.T
    else:
        G = np.random.randn(d0, n)
        Q, _ = np.linalg.qr(G)
    d = np.sqrt(2 * np.random.gamma(d0/2., 1., d0))
    for i in range(Q.shape[0]):
        Q[i, :] *= d[i]

    return Q, None


def generate_rsimplex_weights(k, n, r=None):
    d0 = get_d0(k, n)
    k = int(np.ceil(d0 / 2 / (n+1)))
    if r is None:
        r = radius(n, k)
    S = rnsimp(n)
    Mp = single(3, n)
    M = np.dot(Mp, S).T
    for i in range(1, k):
        Mp = single(3, n)
        L = np.dot(Mp, S)
        M = np.vstack((M, L.T))
    mp = n + 1
    r = np.repeat(r, mp)
    M = np.einsum('i,ij->ij', r, M)
    M = np.vstack((M, -M))
    w = sqrt(n) / r
    w = np.concatenate((w, w))
    return M[:d0, :], w[:d0]


def generate_but_weights(k, n):
    '''
    '''
    c = np.log2(n)
    f = np.floor(c)
    d0 = get_d0(k, n)
    if f != c:
        n = int(2 ** (f+1))
    t = int(np.ceil(d0/n))
    cos, sin, perm = butterfly_params(n, t)
    M = np.zeros((d0, n))

    for i in range(t-1):
        M[i*n:(i+1)*n, :] = butterfly(n, cos[i], sin[i], perm[i])[0]
    i = t-1
    M[i*n:, :] = butterfly(n, cos[i], sin[i], perm[i])[0][:d0 - i*n, :]
    M = np.sqrt(n) * M[:d0, :]

    return M, None


