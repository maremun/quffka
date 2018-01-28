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
from rom import generate_rademacher_weights, generate_gort_weights
from qmc import generate_sobol_weights, generate_halton_weights, \
                generate_lattice_weights


APPROX = OrderedDict({'exact': None,
          'G': [True, generate_random_weights],
          'Gort': [True, generate_orthogonal_weights],
          'ROM': [True, generate_rademacher_weights],
          'H': [True, generate_householder_weights],
          'B dense': [True, generate_butterfly_weights],
          'B': [False, butt_quad_mapping]})
KERNEL = OrderedDict({
             'RBF': rbf_kernel
             'Arccos 0': arccos0_kernel
             'Arccos 1': arccos1_kernel})
KERNEL_C = {'rbf': 1,
            'arccos0': 2,
            'arccos1': 2,
            'linear': 1,
            'angular': 1}
KERNEL_B = {'rbf': lambda w: 1 - np.mean(np.power(w, 2)),
            'arccos0': lambda w: 0.5 * (1 - np.mean(np.power(w, 2))),
            'arccos1': lambda w: 0,
            'linear': lambda w: 0,
            'angular': lambda w: 0}
KERNEL_F = {'rbf': lambda mx: np.vstack((np.cos(mx), np.sin(mx))),
            'arccos0': lambda mx: np.heaviside(mx, 0.5),
            'arccos1': lambda mx: np.maximum(mx, 0.),
            'linear': none,
            'angular': np.sign}
N_APPROX = len(APPROX.keys())


def kernel(X, Y, n, kernel_type, approx_type, **kwargs):
    assert approx_type in APPROX, 'No such approximation type.'
    assert kernel_type in KERNEL, 'No such kernel type.'

    if kernel_type == 'rbf':
        if 'gamma' in kwargs:
            gamma = kwargs['gamma']
        else:
            gamma = 1. / n
        X *= sqrt(2 * gamma)  # sklearn has k_rbf = exp(-gamma||x-y||^2)
        Y *= sqrt(2 * gamma)

    if approx_type == 'exact':
        kernel = KERNEL(kernel_type)[0]
        K = kernel(x, y)
        return K

    c = KERNEL_C[kernel_type]
    b = KERNEL_B[kernel_type]
    D = get_D(d, n)

    Z = np.vstack(X, Y)  # stack X and Y
    Mz, w = mapping(Z, n, kernel_type, approx_type, **kwargs)
    Mx, My = np.split(Mz, [X.shape[0]], 0)
    K = c * np.dot(Mx.T, My) / D
    if w is not None:
        K += b(w)

    return K


def mapping(Z, n, kernel_type, approx_type, **kwargs)
    flag, f = APPROX[approx_type]
    non_linear = KERNEL_F[kernel_type]

    if flag:
        M, w = f(Z, n)
        d1 = M.shape[1]
        d = x.shape[1]
        if d1 > d:
            x = pad_data(x, d1, d)
        MZ = np.dot(M, Z.T)
    else:
        MZ, w = f(Z, n, **kwargs)

    MZ = non_linear(MZ)
    if w is not None:
        MZ = np.einsum('i,ij->ij', w, MZ)

    return MZ, w
