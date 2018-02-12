#!/usr/bin/env python3
#   encoding: utf-8
#   experiment.py
# TODO documentation
import numpy as np

from tqdm import tqdm
from time import perf_counter

from approximate import APPROX, kernel, mapping
from dataset import PARAMS, make_dataset
from performance import error


def relative_errors(X, Y, kernel_type, approx_types, start_deg=1, max_deg=2,
               step=1, shift=1, runs=3):
    K_exact = kernel(X, Y, None, kernel_type, 'exact')
    steps = int(np.ceil(max_deg/step))
    errs = {}
    ranks = {}
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

    return errs


def time(approx_types):
    n = 1
    runs = 3
    times = {}
    for approx_type in tqdm(approx_types):
        print(approx_type)
        times[approx_type] = np.zeros((len(PARAMS.keys()), runs))
        if approx_type == 'exact':
            continue
        for j, dataset_name in enumerate(PARAMS.keys()):
            X, _, _= make_dataset(dataset_name)
            for r in range(runs):
                start_time = perf_counter()
                mapping(X, n, 'RBF', approx_type)
                elapsed = perf_counter() - start_time
                times[approx_type][j, r] = elapsed

    return times
