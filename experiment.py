#!/usr/bin/env python3
#   encoding: utf-8
#   experiment.py
# TODO documentation
import numpy as np
import scipy.stats as st

from tqdm import tqdm

from performance import error
from approximate import APPROX, kernel


def relative_errors(X, Y, kernel_type, approx_types, start_deg=1, max_deg=2,
               step=1, shift=1, runs=3):
    K_exact = kernel(X, Y, None, kernel_type, 'exact')
    steps = int(np.ceil(max_deg/step))
    errs = {}
    ranks = {}
    i = 0
    for approx_type in tqdm(approx_types):
        errs[approx_type] = np.zeros((steps, runs))
        ranks[approx_type] = np.zeros((steps, runs))
        if approx_type == 'exact':
            continue
        for j, deg in enumerate(np.arange(start_deg, max_deg+step, step)):
            nn = deg + shift
            for r in range(runs):
                K = kernel(X, Y, nn, kernel_type, approx_type)
                ranks[approx_type][j, r] = np.linalg.matrix_rank(K)
                errs[approx_type][j, r] = error(K_exact, K)
        i += 1

    return errs
