#!/usr/bin/env python3
#   encoding: utf-8
#   kernels.py
# TODO documentation!
import numpy as np

from math import sqrt

from basics import pad_data


def calculate_theta(x, z):
    nx = np.sqrt(np.einsum('ij,ij->i', x, x))
    nz = np.sqrt(np.einsum('ij,ij->i', z, z))
    nxnz = np.einsum('i,j', nx, nz)
    dot = np.einsum('ik,jk->ij', x, z)
    div = np.einsum('ij,ij->ij', dot, 1. / nxnz)
    div[div >= 1.] = 1.
    div[div <= -1.] = -1.
    theta = np.arccos(div)
    return theta, nxnz


def arccos1_kernel(x, z):
    theta, nxnz = calculate_theta(x, z)
    return (nxnz/np.pi)*(np.sin(theta)+(np.pi - theta)*np.cos(theta))


def approximate_arccos1_kernel(x, z, M, w=None, **kwargs):
    nsamples = M.shape[0]
    n = x.shape[1]
    d1 = M.shape[1]
    if d1 > n:
        x, z = pad_data(x, z, d1, n)
    Mx = np.maximum(np.dot(M, x.T), 0)
    Mz = np.maximum(np.dot(M, z.T), 0)
    if w is not None:
        multiply = lambda z: np.einsum('i,ij->ij', w, z)
        Mx = multiply(Mx)
        Mz = multiply(Mz)
    K = 2 * np.dot(Mx.T, Mz) / nsamples
    return K


def arccos0_kernel(x, z):
    theta, _ = calculate_theta(x, z)
    return 1. - theta/np.pi


def approximate_arccos0_kernel(x, z, M, w=None, **kwargs):
    nsamples = M.shape[0]
    n = x.shape[1]
    d1 = M.shape[1]
    if d1 > n:
        x, z = pad_data(x, z, d1, n)
    Mx = np.heaviside(np.dot(M, x.T), 0.5)
    Mz = np.heaviside(np.dot(M, z.T), 0.5)
    if w is not None:
        multiply = lambda z: np.einsum('i,ij->ij', w, z)
        Mx = multiply(Mx)
        Mz = multiply(Mz)
    K = 2 * np.dot(Mx.T, Mz) / nsamples
    if w is not None:
        K += 0.5 * (1 - np.mean(np.power(w, 2)))
    return K


def approximate_rbf_kernel(x, z, M, w=None, **kwargs):
    '''
    gamma: from kernel params
    '''
    nsamples = M.shape[0] # output dimesion
    n = x.shape[1] # input dimension
    d1 = M.shape[1]
    if d1 > n:
        x, z = pad_data(x, z, d1, n)
    if 'gamma' in kwargs:
        gamma = kwargs['gamma']
    else:
        gamma = 1. / n
    gamma = 1. / n

    sigma = 1.0 / sqrt(2 * gamma)
    def get_rbf_fourier_features(M, x, ww=None):
        Mx = M.dot(x.T) / sigma
        features = np.vstack((np.cos(Mx),
                              np.sin(Mx)))

        if ww is not None:
            features = np.einsum('i,ij->ij', ww, features)
        return features

    ww = None
    if w is not None:
        ww = np.concatenate((w, w))
    Mx = get_rbf_fourier_features(M, x, ww)
    Mz = get_rbf_fourier_features(M, z, ww)
    K = Mx.T.dot(Mz) / nsamples
    if w is not None:
        K += 1 - np.mean(np.power(w, 2))
    return K


def linear_kernel(x, z):
    K = x.dot(z.T)
    return K


def approximate_linear_kernel(x, z, M, w=None, **kwargs):
    nsamples = M.shape[0]
    n = x.shape[1]
    d1 = M.shape[1]
    if d1 > n:
        x, z = pad_data(x, z, d1, n)
    Mx = M.dot(x.T)
    Mz = M.dot(z.T)
    if w is not None:
        multiply = lambda z: np.einsum('i,ij->ij', np.sqrt(w), z)
        Mx = multiply(Mx)
        Mz = multiply(Mz)

    K = np.dot(Mx.T, Mz) / nsamples
    return K


def angular_kernel(x, z):
    theta, _ = calculate_theta(x, z)
    return 1. - 2 * theta/np.pi


def approximate_angular_kernel(x, z, M, w=None, **kwargs):
    nsamples = M.shape[0]
    n = x.shape[1]
    d1 = M.shape[1]
    if d1 > n:
        x, z = pad_data(x, z, d1, n)
    Mx = np.sign(np.dot(M, x.T))
    Mz = np.sign(np.dot(M, z.T))
    if w is not None:
        multiply = lambda z: np.einsum('i,ij->ij', w, z)
        Mx = multiply(Mx)
        Mz = multiply(Mz)
    K = np.dot(Mx.T, Mz) / nsamples
    return K
