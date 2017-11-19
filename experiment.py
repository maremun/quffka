#!/usr/bin/env python3
#   encoding: utf-8
#   experiment.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from time import perf_counter
from timeit import default_timer as timer
from sklearn.metrics.pairwise import rbf_kernel
from kernels import approximate_rbf_kernel, approximate_arccos1_kernel, \
                    approximate_linear_kernel, arccos1_kernel, linear_kernel, \
                    pad_data, approximate_arccos0_kernel, arccos0_kernel, \
                    angular_kernel, approximate_angular_kernel
#from utils import generate_halton_weights, generate_lattice_weights, \
#                  generate_random_weights, generate_sobol_weights, \
#                  generate_householder_weights, generate_orthogonal_weights, \
#                  get_estimator_score
from utils import generate_random_weights, generate_householder_weights, \
                  get_estimator_score, get_d0, generate_halton_weights, \
                  radius, butterfly_params, generate_butterfly_weights, \
                  generate_sobol_weights
from rom import generate_rademacher_weights, generate_gort_weights, \
                generate_rsimplex_weights, generate_but_weights
from tqdm import tnrange
from mapping import fast_batch_approx_arccos1, fast_batch_approx_linear, \
                    fast_batch_approx_rbf, fast_batch_mapping_matvec, \
                    fast_mapping_matvec, fast_batch_approx_arccos0, \
                    fast_batch_approx_angular
from numba import jit, prange

APPROX_TYPES = ['B', 'H',
                'G', 'ROM', 'Gort', 'Halton', 'Sobol']#, 'rsimplex',
#                'butterfly', 'dense B simplex']
#APPROX_TYPES = ['B simplex', 'H simplex', 'sobol', 'halton', 'lattice',
#                'random', 'orf', 'rademacher', 'gort']
KERNEL_TYPES = ['rbf', 'arccos1', 'arccos0', 'linear', 'angular']


def time_embed(approx, sampled_set, k):
    stacked = np.asfortranarray(np.vstack((sampled_set, sampled_set))).T
    n = sampled_set.shape[1]

    if approx == 1:
        n = stacked.shape[0]
        r = radius(n, k)
        b_params = butterfly_params(n, k)
        start_time = perf_counter()
        Mx, _ = fast_batch_mapping_matvec(stacked, k, r, b_params)
    elif approx == 2:
        M, _ = generate_householder_weights(k, n)
        #M, _ = generate_hp_weights(k, n)
        start_time = perf_counter()
        Mx = np.dot(M, sampled_set.T)
    elif approx == 3:
        M, _ = generate_random_weights(k, n)
        start_time = perf_counter()
        Mx = np.dot(M, sampled_set.T)
    elif approx == 4:
        M, _ = generate_rademacher_weights(k, n)
        d1 = M.shape[1]
        if d1 > n:
            sampled_set, _ = pad_data(sampled_set, sampled_set, d1, n)
        start_time = perf_counter()
        Mx = np.dot(M, sampled_set.T)
    elif approx == 5:
        M, _ = generate_gort_weights(k, n)
        start_time = perf_counter()
        Mx = np.dot(M, sampled_set.T)
    else:
        raise Exception('no such approximation type')
    elapsed = perf_counter() - start_time
    return elapsed


def kernel(sampled_set, kernel_type, k=1, approx=False, **kwargs):
    assert approx in range(len(APPROX_TYPES) + 1), \
            'approx: 0 (exact) to the number of available approximation types'
    assert kernel_type in KERNEL_TYPES, 'available kernels %r' % kernel_type

    n = sampled_set.shape[1]
    even = False
    if not approx:
        if kernel_type == 'rbf':
            kernel = rbf_kernel
        elif kernel_type == 'arccos1':
            kernel = arccos1_kernel
        elif kernel_type == 'arccos0':
            kernel = arccos0_kernel
        elif kernel_type == 'linear':
            kernel = linear_kernel
        elif kernel_type == 'angular':
            kernel = angular_kernel
        K = kernel(sampled_set, sampled_set)

    else:
        if kernel_type == 'rbf':
            even = True
            fast_batch_approx = fast_batch_approx_rbf
            kernel = approximate_rbf_kernel
        elif kernel_type == 'arccos1':
            fast_batch_approx = fast_batch_approx_arccos1
            kernel = approximate_arccos1_kernel
        elif kernel_type == 'arccos0':
            even = True
            fast_batch_approx = fast_batch_approx_arccos0
            kernel = approximate_arccos0_kernel
        elif kernel_type == 'linear':
            fast_batch_approx = fast_batch_approx_linear
            kernel = approximate_linear_kernel
        elif kernel_type == 'angular':
            even = True
            fast_batch_approx = fast_batch_approx_angular
            kernel = approximate_angular_kernel

        if approx == 1:
            stacked = np.asfortranarray(np.vstack((sampled_set, sampled_set))).T
            t = 2 * k
            r = radius(n, t)
            b_params = butterfly_params(n, t)
            K = fast_batch_approx(stacked, sampled_set.shape[0], k, r, b_params)
        elif approx == 2:
            M, w = generate_householder_weights(k, n, even=even)
            K = kernel(sampled_set, sampled_set, M, w)
        elif approx == 3:
            M, w = generate_random_weights(k, n)
            K = kernel(sampled_set, sampled_set, M, w)
        elif approx == 4:
            M, w = generate_rademacher_weights(k, n)
            K = kernel(sampled_set, sampled_set, M, w)
        elif approx == 5:
            M, w = generate_gort_weights(k, n)
            K = kernel(sampled_set, sampled_set, M, w)
        elif approx == 6:
            M, w = generate_halton_weights(k, n)
            K = kernel(sampled_set, sampled_set, M, w)
        elif approx == 7:
            #r = radius(n, k)
            M, w = generate_sobol_weights(k, n)
            K = kernel(sampled_set, sampled_set, M, w)
            #M, w = generate_rsimplex_weights(k, n, r)
            #K = kernel(sampled_set, sampled_set, M, w)
        elif approx == 8:
            M, w = generate_butterfly_weights(k, n)
            K = kernel(sampled_set, sampled_set, M, w)
        elif approx == 9:
            M, w = generate_butterfly_weights(k, n)
            K = kernel(sampled_set, sampled_set, M, w)
        else:
            raise Exception('no such approximation type %d' % approx)

    return K


def error(exact, approx):
    err = np.linalg.norm(exact - approx) / np.linalg.norm(exact)

    return err


def mse(exact, approx):
    mse = ((exact - approx)**2).mean()

    return mse


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


def time_vs_features(datasets, runs):
    k = 1.
    times = np.zeros((len(APPROX_TYPES), len(datasets), runs))
    for d, dataset in enumerate(datasets):
        n = dataset.shape[1]
        for a in tnrange(len(APPROX_TYPES)):
            for r in range(runs):
                times[a, d, r] = time_embed(a + 1, dataset, k)

    return times


def run_experiment(kernel_type, sampled_set, params, dataset_name, d):
    start_deg, max_deg, runs, shift, step, NSAMPLES = params
    errs, mses = approximation(sampled_set, kernel_type, start_deg, max_deg,
                               step, shift, runs)

    save_figure_data(dataset_name, kernel_type, params, errs)
    save_figure_data(dataset_name, kernel_type, params, mses, False)
    plot_approx_errors(errs, d, dataset_name, kernel_type, params)
    plt.show()
    plot_approx_errors(mses, d, dataset_name, kernel_type, params, mse=True)
    plt.show()
    plot_approx_errors(errs, d, dataset_name, kernel_type, params, True)
    plt.show()
    plot_approx_errors(mses, d, dataset_name, kernel_type, params, True,
            mse=True)
    plt.show()

    return errs, mses


