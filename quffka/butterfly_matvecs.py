#   encoding: utf-8
#   butterfly.py
# TODO Documentation!

import numpy as np

from numba import jit

NQ = 3


@jit(nopython=True)
def batch_factor_matvec(x, n, cos, sin):
    '''
    Matvec for Q_n^T x, where n is the index number of n'th factor
        of butterfly matrix Q. Facilitates fast butterfly simplex weights
        multiplication by data vector:
        [QV]^T x = V^T Q^T x = V^T Q_{log2(d)}^T ... Q_0^T x

    Args:
    =====
    x: a batch of data vectors
    n: the index number of n'th butterfly factor Q_n
    cos: cosines used to generate butterfly matrix Q
    sin: sines used to generate butterfly matrix Q
    '''
    nobj = x.shape[1]
    N = x.shape[0]
    d = len(cos) + 1
    blockn = 2 ** n
    nblocks = int(np.ceil(N/blockn))
    r = np.copy(x)
    step = blockn // 2
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
    shift = blockn * i
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
def batch_butterfly_matvec(x, cos, sin, p):
    '''
    Apply butterfly matvec NQ times.
    '''
    d = x.shape[0]
    h = int(np.ceil(np.log2(d)))

    for _ in range(NQ):
        for n in range(1, h+1):
            x = batch_factor_matvec(x, n, cos, sin)
        x = x[p, :]
    return x
