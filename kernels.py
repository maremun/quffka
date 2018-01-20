#!/usr/bin/env python3
#   encoding: utf-8
#   kernels.py
# TODO documentation!
import numpy as np

from math import sqrt

from mapping import dense_map_data


# ARC-COSINE
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
    D = M.shape[0]
    f = lambda Mx: np.maximum(Mx, 0.)

    Mx = map_data(x, M, w, f)
    Mz = map_data(z, M, w, f)
    K = 2. * np.dot(Mx.T, Mz) / D

    return K


def arccos0_kernel(x, z):
    theta, _ = calculate_theta(x, z)
    return 1. - theta/np.pi


def approximate_arccos0_kernel(x, z, M, w=None, **kwargs):
    D = M.shape[0]
    f = lambda Mx: np.heaviside(Mx, 0.5)

    Mx = map_data(x, M, w, f)
    Mz = map_data(z, M, w, f)
    K = 2 * np.dot(Mx.T, Mz) / D
    if w is not None:
        K += 0.5 * (1. - np.mean(np.power(w, 2)))

    return K


# GAUSSIAN kernel
def approximate_rbf_kernel(x, z, M, w=None, **kwargs):
    '''
    gamma: from kernel params
    '''
    D = M.shape[0]
    if 'gamma' in kwargs:
        gamma = kwargs['gamma']
    else:
        gamma = 1. / n
    sigma = 1. / sqrt(2. * gamma)
    xs = x / sigma
    zs = z / sigma
    ww = None
    if w is not None:
        ww = np.concatenate((w, w))
    f = lambda Mx: np.vstack((np.cos(Mx), np.sin(Mx))) 

    Mx = map_data(xs, M, ww, f)
    Mz = map_data(zs, M, ww, f)
    K = Mx.T.dot(Mz) / D
    if w is not None:
        K += 1. - np.mean(np.power(w, 2))

    return K


# LINEAR kernel (Gram matrix)
def linear_kernel(x, z):
    K = x.dot(z.T)
    return K


def approximate_linear_kernel(x, z, M, w=None, **kwargs):
    D = M.shape[0]
    MX = map_data(x, M, w, None)
    MZ = map_data(z, M, w, None)
    K = np.dot(Mx.T, Mz) / D

    return K


# ANGULAR kernel
def angular_kernel(x, z):
    theta, _ = calculate_theta(x, z)
    return 1. - 2 * theta/np.pi


def approximate_angular_kernel(x, z, M, w=None, **kwargs):
    D = M.shape[0]
    MX = map_data(x, M, w, np.sign)
    MZ = map_data(z, M, w, np.sign)
    K = np.dot(Mx.T, Mz) / D

    return K
