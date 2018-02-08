#!/usr/bin/env python3
#   encoding: utf-8
#   rom.py
# TODO documentation!
import numpy as np

from math import sqrt
from numba import jit

from basics import get_D, rnsimp, radius
from butterfly import butterfly, butterfly_params


@jit(nopython=True)
def hadamard(d):
    if d < 1:
        lg2 = 0
    else:
        lg2 = int(np.log2(d))
    if 2 ** lg2 != d:
        raise ValueError("d must be an positive integer, and d must be "
                         "a power of 2")

    H = np.zeros((d, d))
    H[0, 0] = 1

    # Sylvester's construction
    for i in range(0, lg2):
        p = 2**i
        H[:p, p:2*p] = H[:p, :p]
        H[p:2*p, :p] = H[:p, :p]
        H[p:2*p, p:2*p] = -H[:p, :p]
    H /= sqrt(d)
    return H


@jit(nopython=True)
def diagonal(d):
    '''
    Generates diagonal matrix D if size n x n with iid Rademacher random
        variables on the diagonal.
    '''
    diag = (np.random.randint(0, 2, size=d) - 0.5) * 2
    D = np.diag(diag)
    return D


@jit(nopython=True)
def single(p, d):
    S = hadamard(d)
    D = diagonal(d)
    M = np.dot(S, D)
    for _ in range(p-1):
        D = diagonal(d)
        M = np.dot(M, np.dot(S, D))
    return M


@jit(nopython=True)
def generate_rademacher_weights(d, n, p=3):
    '''
    Generates n x n S-Rademacher random matrix as in
        https://arxiv.org/abs/1703.00864.
    '''
    c = np.log2(d)
    f = np.floor(c)
    D = get_D(d, n)
    if f != c:
        d = int(2 ** (f + 1))

    M = np.zeros((D, d))

    t = int(np.ceil(D/d))
    for i in range(t-1):
        M[i*d:(i+1)*d, :] = single(p, d)
    i = t - 1
    M[i*d:, :] = single(p, d)[:D - i*d, :]
    M = np.sqrt(d) * M[:D, :]
    return M, None


@jit(nopython=True)
def generate_orthogonal_weights(d, n):
    D = get_D(d, n)
    if D < d:
        G = np.random.randn(d, D)
        Q, _ = np.linalg.qr(G)
        Q = Q.T
    else:
        G = np.random.randn(D, d)
        Q, _ = np.linalg.qr(G)
    d = np.sqrt(2 * np.random.gamma(D/2., 1., D))
    for i in range(Q.shape[0]):
        Q[i, :] *= d[i]
    return Q, None


def generate_rsimplex_weights(d, n, r=None):
    D = get_D(d, n)
    n = int(np.ceil(D/2/(d+1)))
    if r is None:
        r = radius(d, n)
    S = rnsimp(d)
    Mp = single(3, d)
    M = np.dot(Mp, S).T
    for i in range(1, n):
        Mp = single(3, d)
        L = np.dot(Mp, S)
        M = np.vstack((M, L.T))
    mp = d + 1
    r = np.repeat(r, mp)
    M = np.einsum('i,ij->ij', r, M)
    M = np.vstack((M, -M))
    w = sqrt(d) / r
    w = np.concatenate((w, w))
    return M[:D, :], w[:D]


def generate_but_weights(d, n):
    '''
    '''
    c = np.log2(d)
    f = np.floor(c)
    D = get_D(d, n)
    if f != c:
        n = int(2**(f+1))
    t = int(np.ceil(D/d))
    cos, sin, perm = butterfly_params(d, t)
    M = np.zeros((D, d))

    for i in range(t-1):
        M[i*d:(i+1)*d, :] = butterfly(d, cos[i], sin[i], perm[i])[0]
    i = t - 1
    M[i*d:, :] = butterfly(d, cos[i], sin[i], perm[i])[0][:D - i*d, :]
    M = np.sqrt(d) * M[:D, :]
    return M, None
