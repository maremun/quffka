#   econding: utf-8
#   householder.py
# TODO documentation!
from math import sqrt
import numpy as np

from basics import get_d0


@jit(nopython=True)
def householder_reflect(S):
    n = S.shape[0]
    X = np.random.randn(n, n)
    X /= np.linalg.norm(X)
    Q = np.linalg.qr(X)[0]
    return np.dot(Q, S)


def generate_householder_weights(k, n, r=None, even=False):
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
