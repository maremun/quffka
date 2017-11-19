#!/usr/bin/env python3
#   encoding: utf-8
#   butterfly.py


import numpy as np
from numba import jit, prange
from scipy.linalg import block_diag

NQ = 3

@jit(nopython=True)
def butterfly_generating_vector(n):
    '''
    Generates a vector `u` used to construct random butterfly orthogonal
        matrices.

    Args:
    =====
    n: size of generating vector, n = N + 1 to construct random
        butterfly orthogonal matrix of size N x N.

    Returns:
    ========
    u: generating vector used to calculate angles for random butterfly
        orthogonal matrices.
    '''
    l = n//2 - 1
    r = np.random.rand(n-1)
    u = np.zeros(n)

    for i in range(l):
        m = n - 2*i
        s = np.sin(2 * np.pi * r[m-2])
        c = np.cos(2 * np.pi * r[m-2])
        pos = n - 2*np.arange(1, i+1) - 1
        ds = 1. / (pos + 1)
        p = np.prod(r[pos]**ds)

        u[m - 1] = np.sqrt(1. - r[m-3]**(2./(m-3))) * p * s
        u[m - 2] = np.sqrt(1. - r[m-3]**(2./(m-3))) * p * c

    s = np.sin(2 * np.pi * r[0])
    c = np.cos(2 * np.pi * r[0])
    pos = n - 2*np.arange(1, l+1) - 1
    ds = 1. / (pos + 1)

    p = np.prod(r[pos]**ds)
    if n % 2 == 0:
        u[0] = c * p
        u[1] = s * p
    else:
        u[2] = (2 * r[1] - 1) * p
        u[1] = 2 * np.sqrt(r[1] * (1 - r[1])) * p * s
        u[0] = 2 * np.sqrt(r[1] * (1 - r[1])) * p * c

    return u   
    #u = np.random.randn(n)
    #u /= np.linalg.norm(u)

    #return u


@jit(nopython=True)
def butterfly_angles(u):
    '''
    Computes angles (in radians) from components of the generating vector
        `u` for random butterfly orthogonal matrices.

    Args:
    =====
    u: a generating vector for random butterfly orthogonal matrices, use
        make_generating_vector() to obtain one.

    Returns:
    ========
    thetas: an 1-D array of angles for computing random butterfly orthogonal
        matrices.
    '''
    thetas = np.arctan2(u[:-1], u[1:])
    return thetas

#@jit(nopython=True)
def butterfly_block(n, cos, sin):
    '''
    Generates n x n block of a butterfly matrix.
    '''
    i = n//2
    sdiag = np.repeat(sin[i-1], i)
    Q = np.diagflat(-1 * sdiag, i)
    Q -= Q.T
    np.fill_diagonal(Q, cos[i-1])
    return Q

#@jit(nopython=True)
def butterfly_factors(n, cos, sin, N):
    '''
    Generates a sequence of log_2(n) factors for butterfly orthogonal matrix of
         size n x n.

    Args:
    =====
    n: the next power of two in case the desired size N is not one.
    cos: cosines of generating angles.
    sin: sines of generating angles.
    N: the size of butterfly matrix.

    Returns:
    ========
    Qs: sequence of log_2(n) random butterfly orthogonal matrix factors, each
        of size n x n.
    '''
    if n == 1:
        return np.array(1)
    c = np.log2(n)
    f = np.ceil(c)
    if c != f:
        raise Exception('n is not power of two.')

    Qs = []
    for i in range(int(f)):
        blockn = 2**(i+1)
        nblocks = n//blockn
        blocks = [butterfly_block(blockn, cos[blockn*j:], sin[blockn*j:]) \
                  for j in range(nblocks)]

        Qs.append(block_diag(*blocks))
        l = (N // blockn) * blockn
        h = l + blockn
        if N > (h - blockn//2) and N < h:
            j = blockn//2 - h + N
            ll = l+j
            hh = -j
            di = np.arange(N+hh-ll) + ll
            Qs[i][di, di] = 1.
            #di = np.diag_indices(N)
            #Qs[i][di[0][ll:hh], di[1][ll:hh]] = 1.

    return Qs

@jit(nopython=True)
def cos_sin(N):
    c = np.log2(N)
    f = np.ceil(c)
    n = int(2 ** f)
    u = butterfly_generating_vector(n)
    thetas = butterfly_angles(u)
    if c != f:
        thetas = np.concatenate((thetas[:N-1], np.array([0.] * (n - N))))
    cos = np.cos(thetas)
    sin = np.sin(thetas)
    return cos, sin

#@jit(nopython=True)
def butterfly(N, cos=None, sin=None, perm=None):
    '''
    Generates dense random butterfly orthogonal matrix from its factors.
    Args:
    =====
    N: size of the matrix, should be a power of 2.

    Returns:
    ========
    QP: random butterfly orthogonal matrix of size N x N: QPQP...QP (product of
        NQ QP matrices).
    Qs: butterfly factors
    cs: a tuple of cosines and sines
    perm: permutation
    '''
    if N == 1:
        return np.array(1)
    if cos is not None:
        cs = (cos, sin)
    else:
        cs = cos_sin(N)
        cos, sin = cs
        perm = np.arange(N)
        np.random.shuffle(perm)

    n = len(cos) + 1
    Qs = butterfly_factors(n, cos, sin, N)
    Q = np.eye(N)
    Qs = [q[:N, :N] for q in Qs]
    for q in Qs:
        Q = Q.dot(q)
    QP = Q[:, perm]
    for _ in range(NQ - 1):
        QP = QP.dot(Q[:, perm])

    return QP, Qs, cs, perm

def dense_butterfly(n, cos, sin, hat=False):
    '''
    Generates dense random butterfly orthogonal matrix.

    Args:
    =====
    n: resulting matrix is of size n x n.
    cos: cosines of generating angles.
    sin: sines of generating angles.

    Returns:
    ========
    Q: random butterfly orthogonal matrix of size n x n.
    '''
    if n == 1:
        return np.array(1)
    c = np.log2(n)
    if c != np.floor(c):
        raise Exception('n is not power of two.')

    h = n//2
    Qhalf = dense_butterfly(h, cos, sin)
    Qhalf_hat = dense_butterfly(h, cos[h:], sin[h:], n)
    i = n//2 - 1
    if not hat:
        Qn = np.hstack((Qhalf * cos[i],
                        -Qhalf * sin[i]))
        Qn_hat = np.hstack((Qhalf_hat * sin[i],
                            Qhalf_hat * cos[i]))
    else:
        Qn = np.hstack((Qhalf * cos[i],
                        -Qhalf * sin[i]))
        Qn_hat = np.hstack((Qhalf_hat * sin[i],
                            Qhalf_hat * cos[i]))
    Qn = np.vstack((Qn, Qn_hat))
    return Qn

#@jit(nopython=True)
def butterfly_transform(S, cos, sin, perm):
    '''
    Naive implementation of simplexes randomly rotated by butterfly matrix.

    Args:
    =====
    S: a matrix [n x n+1] of a random n-simplex.

    Returns:
    ========
    QS: QS, where Q is random butterfly matrix.
    '''
    N = S.shape[0]
    if cos is not None:
        Q, _, _, _ = butterfly(N, cos, sin, perm)
    else:
        Q, _, _, _ = butterfly(N)
    QS = Q.dot(S)
    return QS


@jit(nopython=True)
def factor_matvec(x, k, cos, sin):
    '''
    Matvec for Q_k^T x, where k is the index number of k'th factor
        of butterfly matrix Q. Facilitates fast butterfly simplex weights
        multiplication by data vector:
        [QV]^T x = V^T Q^T x = V^T Q_{log2(n)}^T ... q_0^T x

    Args:
    =====
    x: data vector
    k: the index number of k'th butterfly factor Q_k
    cos: cosines used to generate butterfly matrix Q
    sin: sines used to generate butterfly matrix Q
    '''
    n = len(cos) + 1
    N = x.shape[0]
    blockn = 2**k
    nblocks = int(np.ceil(N/blockn))
    step = blockn//2
    for i in range(nblocks - 1):
        shift = blockn*i
        idx = step + shift - 1
        c = cos[idx]
        s = sin[idx]
        for j in range(step):
            i1 = shift + j
            i2 = shift + j + step
            y1 = x[i1]
            y2 = x[i2]
            x[i1] = c * y1 + s * y2
            x[i2] = -s * y1 + c * y2

    # Last block is special since N might not be a power of 2,
    # which causes cutting the matrix and replacing some elements
    # with ones.
    # We calculate t - the number of lines to fill in before proceeding.
    i = nblocks - 1
    shift = blockn*i
    idx = step + shift - 1
    c = cos[idx]
    s = sin[idx]
    t = N - shift - step
    for j in range(t):
            i1 = shift + j
            i2 = shift + j + step
            y1 = x[i1]
            y2 = x[i2]
            x[i1] = c * y1 + s * y2
            x[i2] = -s * y1 + c * y2

    return x

@jit(nopython=True)
def batch_factor_matvec(x, k, cos, sin):
    '''
    Matvec for Q_k^T x, where k is the index number of k'th factor
        of butterfly matrix Q. Facilitates fast butterfly simplex weights
        multiplication by data vector:
        [QV]^T x = V^T Q^T x = V^T Q_{log2(n)}^T ... Q_0^T x

    Args:
    =====
    x: a batch of data vectors
    k: the index number of k'th butterfly factor Q_k
    cos: cosines used to generate butterfly matrix Q
    sin: sines used to generate butterfly matrix Q
    '''
    nobj = x.shape[1]
    N = x.shape[0]
    n = len(cos) + 1
    blockn = 2**k
    nblocks = int(np.ceil(N/blockn))
    r = np.copy(x)
    step = blockn//2
    for i in range(nblocks - 1):
        shift = blockn*i
        idx = step + shift - 1
        c = cos[idx]
        s = sin[idx]
        for j in range(step):
            i1 = shift + j
            i2 = i1 + step
            for o in range(nobj):
                y1 = x[i1, o]
                y2 = x[i2, o]
                r[i1, o] = c * y1 + s * y2
                r[i2, o] = -s * y1 + c * y2

    # Last block is special since N might not be a power of 2,
    # which causes cutting the matrix and replacing some elements
    # with ones.
    # We calculate t - the number of lines to fill in before proceeding.
    i = nblocks - 1
    shift = blockn*i
    idx = step + shift - 1
    c = cos[idx]
    s = sin[idx]
    t = N - shift - step
    for j in range(t):
        i1 = shift + j
        i2 = i1 + step
        for o in range(nobj):
            y1 = x[i1, o]
            y2 = x[i2, o]
            r[i1, o] = c * y1 + s * y2
            r[i2, o] = -s * y1 + c * y2

    return r

@jit(nopython=True)
def butterfly_matvec(x, cos, sin, p):
    n = x.shape[0]
    h = int(np.ceil(np.log2(n)))
    for _ in range(NQ):
        for k in range(1, h+1):
            x = factor_matvec(x, k, cos, sin)
        x = x[p]

    return x

@jit(nopython=True)
def batch_butterfly_matvec(x, cos, sin, p):
    n = x.shape[0]
    h = int(np.ceil(np.log2(n)))

    for _ in range(NQ):
        for k in range(1, h+1):
            x = batch_factor_matvec(x, k, cos, sin)
        x = x[p, :]

    return x
