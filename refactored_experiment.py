#   encoding: utf-8
#   experiment.py

import numpy as np
import scipy.stats as st

from numba import jit, prange
from sklearn.metrics.pairwise import rbf_kernel
from tqdm import tnrange

from kernels import approximate_arccos0_kernel, approximate_arccos1_kernel, \
                    approximate_rbf_kernel, arccos0_kernel, arccos1_kernel
from mapping import fast_batch_approx_arccos0, fast_batch_approx_arccos1, \
                    fast_batch_approx_rbf
from utils import generate_householder_weights, generate_random_weights, \
                  get_estimator_score, get_d0, \
                  radius, butterfly_params, generate_butterfly_weights, \
                  generate_sobol_weights, pad_data
from rom import generate_rademacher_weights, generate_gort_weights, \
                generate_rsimplex_weights, generate_but_weights


APPROX = {'exact': None,
          'G': [False, generate_random_weights],
          'Gort': [False, generate_orthogonal_weights],
          'ROM': [False, generate_rademacher_weights],
          'H': [False, generate_householder_weights],
          'B dense': [False, generate_butterfly_weights],
          'B': [True, None]}
KERNEL = {'RBF': [rbf_kernel, approximate_rbf_kernel, fast_batch_approx_rbf]
          'Arccos 0': [arccos0_kernel, approximate_arccos0_kernel,
                       fast_batch_approx_arccos0]
          'Arccos 1': [arccos1_kernel, approximate_arccos1_kernel,
                       fast_batch_approx_arccos1]}


def kernel(x, y, k, kernel_type, approx_type):
    assert approx_type in APPROX, 'No such approximation type.'
    assert kernel_type in KERNEL, 'No such kernel type.'

    if approx_type == 'exact':
        kernel = KERNEL(kernel_type)[0]
        K = kernel(x, y)
        return K

    flag, generate_weights = APPROX[approx_type]
    kernel = KERNEL[kernel_type][flag+1]
    if flag:
        M, w = generate_weights(k, n)
    K = kernel(x, y, M, w)

    return K


def error(exact, approx):
    err = np.linalg.norm(exact - approx) / np.linalg.norm(exact)

    return err


def approximation(sampled_set, kernel_type, start_deg=1, max_deg=3,
                  step=1, shift=1, runs=3):
    K_exact = kernel(sampled_set, kernel_type)
    print('exact rank', np.linalg.matrix_rank(K_exact))
    steps = int(np.ceil(max_deg/step))
    errs = np.zeros((len(APPROX_TYPES), steps, runs))
    mses = np.zeros((len(APPROX_TYPES), steps, runs))
    ranks = np.zeros((len(APPROX_TYPES), steps, runs))
    #for i in tnrange(len(APPROX_TYPES)):
    for t in tnrange(1):
        i = 6
        print(APPROX_TYPES[i])
        for j, deg in enumerate(np.arange(start_deg, max_deg+step, step)):
            k = deg + shift
            for r in range(runs):
                K = kernel(sampled_set, kernel_type, k, i+1)
                ranks[i, j, r] = np.linalg.matrix_rank(K)
                errs[i, j, r] = error(K_exact, K)
                mses[i, j, r] = mse(K_exact, K)
        print(ranks[i, :, :].mean(1))

    return errs, mses


