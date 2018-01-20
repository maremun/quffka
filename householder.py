#   econding: utf-8
#   householder.py
# TODO documentation!
import numpy as np

from math import sqrt

from basics import get_D


@jit(nopython=True)
def householder_reflect(S):
    d = S.shape[0]
    X = np.random.randn(d, d)
    X /= np.linalg.norm(X)
    Q = np.linalg.qr(X)[0]
    return np.dot(Q, S)


def generate_householder_weights(d, n, r=None, even=False):
    D = get_D(d, n)
    M = np.empty((D, n))
    if even:
        t = int(np.ceil(2*n))
    else:
        t = int(np.ceil(n))
    if r is None:
        r = radius(d, t)

    w = np.empty(D)
    S = rnsimp(d)

    if n < 1:
        if n > 0.5:
            M[:d+1, :] = r[0] * householder_reflect(S).T
            w[:] = sqrt(d) / r[0]
            if even:
                M[d+1:, :] = r[1] * householder_reflect(S).T[:D-(d+1), :]
                w[d+1:] = sqrt(d) / r[1]
            else:
                M[d+1:, :] = -M[:D-(d+1), :]
        else:
            M = r[0] * householder_reflect(S).T[:D, :]
            w[:] = sqrt(d) / r[0]
        return M, w

    if even:
        for i in range(t-1):
            M[i*(d+1):(i+1)*(d+1), :] = r[i] * householder_reflect(S).T
            w[i*(d+1):(i+1)*(d+1)] = sqrt(d) / r[i]
        i = t-1
        M[i*(d+1):, :] = r[i] * householder_reflect(S).T[:D-(t-1)*(d+1)]
        w[i*(d+1):] = sqrt(d) / r[i]
    else:
        for i in range(t):
            M[i*(d+1):(i+1)*(d+1), :] = r[i] * householder_reflect(S).T
            w[i*(d+1):(i+1)*(d+1)] = sqrt(d) / r[i]
        div = t*(d+1)
        M[div:, :] = -M[:D-div, :]
        w[div:] = w[:D-div]
    return M, w
