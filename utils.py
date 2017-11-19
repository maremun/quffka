#!/usr/bin/env python3
#   encoding: utf-8
#   utils.py


import numpy as np
from math import sqrt
from numba import jit, prange, njit
#from scipy.special import gamma
from scipy.stats import chi, norm
from sklearn import svm, metrics
from butterfly import cos_sin, butterfly, butterfly_transform
#from lattice import lattice
from sobol import i4_sobol_generate
import ghalton

EPS = 1e-6

@jit(nopython=True)
def radius(n, k):
    h = int(np.ceil(k))
    rm = n + 2
    r = np.sqrt(2 * np.random.gamma(rm/2., 1., h))
    return r

@jit(nopython=True)
def butterfly_params(n, k):
    h = int(np.ceil(k))
    log = np.log2(n)
    next_power = 2**int(np.ceil(log))
    cos = np.empty((h, next_power-1))
    sin = np.empty((h, next_power-1))
    perm = np.empty((h, n), np.int32)
    for i in range(h):
        c, s = cos_sin(n)
        cos[i] = c
        sin[i] = s
        p = np.arange(n)
        np.random.shuffle(p)
        perm[i] = p

    return cos, sin, perm

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

 #   for o in range(nobj):
  #      s = 0.
   #     for i in range(n):
    #        rv[i] = sqrt(mp / ((n - i) * n * (n - i + 1.))) * x[o, i]
     #       r[o, i] = s + rv[i] * (n-i)
      #      s += -rv[i]
       # r[o, n] = s

    return r

@jit(nopython=True)
def householder_reflect_old(S):
    '''
    Returns:
    ========
    S: m x m+1 matrix
    '''
    M = np.copy(S)
    m = S.shape[0]
    #multiply by random orthogonal matrix
    x = np.random.randn(m+1)
    for k in range(m-2, -1, -1):
        al = 0.
        for i in range(k, m):
            al += x[i]**2
        al = -sqrt(al)
        bt = 1. / (al * (al + x[k]))
        x[k] += al
        for j in range(k, m + 1):
            al = 0.
            for i in range(k, m):
                al += x[i] * M[i, j]
            al = bt * al
            for i in range(k, m):
                M[i, j] -= x[i] * al

    return M

@jit(nopython=True)
def householder_reflect(S):
    n = S.shape[0]
    X = np.random.randn(n, n)
    X /= np.linalg.norm(X)
    Q = np.linalg.qr(X)[0]

    return np.dot(Q, S)

@jit(nopython=True)
def get_d0(k, n):
    '''
    when using simplex method to generate points, we add also reflected ones,
    so the overall number of points created is 2 * (n+1).
    '''
    d0 = int(2 * k * (n+1))

    return d0


#@jit(nopython=True)
def generate_butterfly_weights(k, n, r=None, b_params=None, even=False):
    d0 = get_d0(k, n)
    if even:
        t = int(np.ceil(2*k))
    else:
        t = int(np.ceil(k))
    if r is None:
        r = radius(n, k)
    if b_params is None:
        b_params = butterfly_params(n, k)
    S = rnsimp(n)
    cos, sin, perm = b_params
    M = butterfly_transform(S, cos[0], sin[0], perm[0]).T
    for i in range(1, t):
        L = butterfly_transform(S, cos[i], sin[i], perm[i])
        M = np.vstack((M, L.T))
    mp = n + 1
    r = np.repeat(r, mp)
    M = np.einsum('i,ij->ij', r, M)
    w = sqrt(n) / r
    if even is False:
        M = np.vstack((M, -M))
        w = np.concatenate((w, w))

    return M[:d0, :], w[:d0]


#@jit(nopython=True)
def generate_householder_weights(k, n, r=None, even=False):
#TODO use simplex matvec to speedup Mx when M is householder weights
    d0 = get_d0(k, n)
    M = np.empty((d0, n))
    if even:
        t = int(np.ceil(2*k))
    else:
        t = int(np.ceil(k))
    if r is None:
        r = radius(n, t)

    w = np.empty(d0)
    S = rnsimp(n)

    if k < 1:
        if k > 0.5:
            M[:n+1, :] = r[0] * householder_reflect(S).T
            w[:] = sqrt(n) / r[0]
            if even:
                M[n+1:, :] = r[1] * householder_reflect(S).T[:d0-(n+1), :]
                w[n+1:] = sqrt(n) / r[1]
            else:
                M[n+1:, :] = -M[:d0-(n+1), :]
                
        else:
            M = r[0] * householder_reflect(S).T[:d0, :]
            w[:] = sqrt(n) / r[0]
        return M, w

    if even:
        for i in range(t-1):
            M[i*(n+1):(i+1)*(n+1), :] = r[i] * householder_reflect(S).T
            w[i*(n+1):(i+1)*(n+1)] = sqrt(n) / r[i]

        i = t-1
        M[i*(n+1):, :] = r[i] * householder_reflect(S).T[:d0-(t-1)*(n+1)]
        w[i*(n+1):] = sqrt(n) / r[i]

    else:
        for i in range(t):
            M[i*(n+1):(i+1)*(n+1), :] = r[i] * householder_reflect(S).T
            w[i*(n+1):(i+1)*(n+1)] = sqrt(n) / r[i]

        div = t*(n+1)
        M[div:, :] = -M[:d0-div, :]
        w[div:] = w[:d0-div]

    return M, w

def generate_lattice_weights(k, n):
    pass
    #d0 = get_d0(k, n)
    #z = lattice[:n]
    #points = (np.outer(range(d0), z) % d0) / d0 + EPS
    #M = norm.ppf(points)
    #return M, None

def generate_sobol_weights(k, n):
    #pass
    d0 = get_d0(k, n)
    points = np.float64(i4_sobol_generate(n, d0, 0)) + EPS
    M = norm.ppf(points).T
    return M, None

def generate_halton_weights(k, n):
    #pass
    d0 = get_d0(k, n)
    sequencer = ghalton.GeneralizedHalton(n)
    points = np.array(sequencer.get(d0))
    M = norm.ppf(points)
    return M, None

def generate_orthogonal_weights(k, n):
    d0 = get_d0(k, n)
    t = int(np.ceil(d0/n))
    Q, _ = butterfly(n)
    d = chi.rvs(n, size=n)
    S = np.diag(d)
    M = S.dot(Q)
    for _ in range(t-1):
        Q, _ = butterfly(n)
        d = chi.rvs(n, size=n)
        S = np.diag(d)
        M = np.vstack((M, S.dot(Q)))

    return M[:d0, :], None


def generate_random_weights(k, n):
    d0 = get_d0(k, n)
    M = np.random.randn(d0, n)
    return M, None


def get_estimator_score(data, labels, kernel_type, task):
    '''
    Args:
    =====
    task: if 0 - SVM, 1 - SVR
    kernel_type: any of sklearn's allowed kernel types.
    data: a tuple of matrices; if kernel_type = 'precomputed',
        data[0] is a train set kernel, data[1] - test set kernel,
        otherwise, data[0] is just a train set, data[1] - test set.
    labels: a tuple of target values, labels[0] is train targets,
        labels[1] - test targets.
    '''
    if task == 0:
        estimator = svm.SVC(kernel=kernel_type)
    elif task == 1:
        estimator = svm.SVR(kernel=kernel_type)
    else:
        raise Exception('task=0 for SVM, task=1 for SVR, \
                         other models not implemented')
    estimator.fit(data[0], labels[0])
    predicted = estimator.predict(data[1])
    if task == 0:
        return metrics.accuracy_score(labels[1], predicted)
    else:
        return metrics.r2_score(labels[1], predicted)
