#!/usr/bin/env python3
#   encoding: utf-8
#   experiment.py
# TODO documentation
import numpy as np

from tqdm import tqdm

from performance import error, mserror
from approximate import APPROX, N_APPROX, kernel


def relative_errors(X, Y, n, kernel_type, start_deg=1, max_deg=3,
               step=1, shift=1, runs=3):
    K_exact = kernel(X, Y, n, kernel_type, 'exact')
    steps = int(np.ceil(max_deg/step))
    errs = np.zeros((N_APPROX, steps, runs))
    mses = np.zeros((N_APPROX, steps, runs))
    ranks = np.zeros((N_APPROX, steps, runs))
    i = 0
    for approx_type in tqdm(APPROX.keys()):
        if approx_type == 'exact':
            continue
        for j, deg in enumerate(np.arange(start_deg, max_deg+step, step)):
            nn = deg + shift
            for r in range(runs):
                K = kernel(X, Y, nn, kernel_type, approx_type)
                ranks[i, j, r] = np.linalg.matrix_rank(K)
                errs[i, j, r] = error(K_exact, K)
                mses[i, j, r] = mserror(K_exact, K)
        i += 1

    return errs, mses
