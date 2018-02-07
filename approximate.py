#!/usr/bin/env python3
#   encoding: utf-8
#   approximate.py
# TODO documentation
import numpy as np

from collections import OrderedDict
from math import sqrt
from sklearn.metrics.pairwise import rbf_kernel

from basics import generate_random_weights, get_D
from butterfly import generate_butterfly_weights
from householder import generate_householder_weights
from kernels import arccos0_kernel, arccos1_kernel
from mapping import butt_quad_mapping
from rom import generate_rademacher_weights, generate_orthogonal_weights
from qmc import generate_halton_weights, generate_lattice_weights

import matplotlib.pyplot as plt

APPROX = OrderedDict({'exact': None,
          #'G': [True, generate_random_weights],
          #'Gort': [True, generate_orthogonal_weights],
          'ROM': [True, generate_rademacher_weights],
          #'H': [True, generate_householder_weights],
          #'B dense': [True, generate_butterfly_weights],
          'B': [False, butt_quad_mapping]})
KERNEL = OrderedDict({
             'RBF': rbf_kernel,
             'Arccos 0': arccos0_kernel,
             'Arccos 1': arccos1_kernel})
KERNEL_C = {'RBF': 1,
            'Arccos 0': 2,
            'Arccos 1': 2,
            'Linear': 1,
            'Angular': 1}
KERNEL_B = {'RBF': lambda w: 1 - np.mean(np.power(w, 2)),
            'Arccos 0': lambda w: 0.5 * (1 - np.mean(np.power(w, 2))),
            'Arccos 1': lambda w: 0,
            'Linear': lambda w: 0,
            'Angular': lambda w: 0}
KERNEL_F = {'RBF': lambda mx: np.hstack((np.cos(mx), np.sin(mx))),
            'Arccos 0': lambda mx: np.heaviside(mx, 0.5),
            'Arccos 1': lambda mx: np.maximum(mx, 0.),
            'Linear': None,
            'Angular': np.sign}
N_APPROX = len(APPROX.keys())


def kernel(X, Y, n, kernel_type, approx_type, **kwargs):
    assert approx_type in APPROX, 'No such approximation type.'
    assert kernel_type in KERNEL, 'No such kernel type.'

    d = X.shape[1]
    Xc = X.copy()
    Yc = Y.copy()
    if kernel_type == 'RBF':
        if 'gamma' in kwargs:
            gamma = kwargs['gamma']
        else:
            gamma = 1. / d
        Xc *= sqrt(2 * gamma)  # sklearn has k_rbf = exp(-gamma||x-y||^2)
        Yc *= sqrt(2 * gamma)

    if approx_type == 'exact':
        kernel = KERNEL[kernel_type]
        K = kernel(Xc, Yc)
        return K

    D = get_D(d, n)
    c = KERNEL_C[kernel_type]
    b = KERNEL_B[kernel_type]

    Z = np.vstack((Xc, Yc))  # stack X and Y
    Mz, w = mapping(Z, n, kernel_type, approx_type)  # , **kwargs) input scaled
    Mx, My = np.split(Mz, [X.shape[0]], 0)
    K = c * np.dot(Mx, My.T) / D
    if w is not None:
        K += b(w)

    return K


def mapping(Z, n, kernel_type, approx_type, **kwargs):
    flag, f = APPROX[approx_type]
    non_linear = KERNEL_F[kernel_type]

    if flag:
        d = Z.shape[1]
        M, w = f(d, n)
        d1 = M.shape[1]
        if d1 > d:
            Z = pad_data(Z, d1, d)
        MZ = np.dot(Z, M.T)
    else:
        MZ, w = f(Z.T, n, **kwargs)
        MZ = MZ.T

    MZ = non_linear(MZ)
    if w is not None:
        if kernel_type == 'RBF':
            w = np.concatenate((w, w))
        MZ = np.einsum('j,ij->ij', w, MZ)

    return MZ, w
