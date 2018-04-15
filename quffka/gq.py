#   enconding: utf-8
#   gaussquad.py
# TODO fix Documentation

import numpy as np
import numpy.polynomial.hermite as herm

from .basics import get_D


def generate_gq_weights(d, n, deg=2):
    '''
    Subsample points and weights for Gauss-Hermite quadrature.
    Args
        d: the number of dimensions of the integral.
        n: number of SR rules to be equivalent to in terms of output feature
           dimensionality D.
        deg: the degree of the polynomial that is approximated accurately by
             this quadrule is 2-1.

    Returns
        W_subsampled: the points, matrix of size d x D.
        A_subsampled: the weights, vector of size D.
    '''
    W, A = herm.hermgauss(deg)
    # Use ortho for higher dimensions
#     W, A = orth.line.schemes.hermite(d, 30)
    # make sure we are not working with object datatype
#     W = np.array(W, dtype='float64')
#     A = np.array(A, dtype='float64')
    # normalize weights
    # this is where the 1/sqrt(pi) constant is hidden
    # since sum of the weights is exactly the constant!
    A = A / A.sum()
    D = get_D(d, n)
    c = np.empty((d, D))
    I = np.arange(W.shape[0])
    for i in range(d):
        c[i] = np.random.choice(I, D, True, A)
    c = c.astype('int')
    W_subsampled = W[c]
    A_subsampled = np.sum(np.log(A[c]), axis=0)
    A_subsampled -= A_subsampled.max()
    A_subsampled = np.exp(A_subsampled)

#    I = np.arange(W.shape[0])
#    c = np.random.choice(I, d*L, True, A)
#    W_subsampled = np.reshape(W[c], (d,L))
#    A_subsampled = np.prod(np.reshape(A[c], (d,L)), axis=0)
##     W_subsampled = np.ascontiguousarray(W_subsampled.T).T  # for faster tensordot

    # normalize weights of subsampled points
    A_subsampled /= A_subsampled.sum()
    W_subsampled *= np.sqrt(2)
    return W_subsampled.T, np.sqrt(A_subsampled)


def explicit_map(Z, W):
    nsamples = W.shape[0]
    ZW = Z @ W
    fZ = np.hstack((np.cos(ZW), np.sin(ZW)))
    return fZ


def rbf_kernel_via_explicit_map(X, Y, A):
    A = np.sqrt(A)
    A = np.concatenate((A, A))
    X = np.einsum('j,ij->ij', A, X)
    Y = np.einsum('j,ij->ij', A, Y)
    K = X @ Y.T
    return K
