#   encoding: utf-8
#   experiment.py
# TODO documentation

import numpy as np

from tqdm import tqdm

from .approximate import APPROX, kernel, mapping
from .basics import radius
from .butterfly import butterfly_params
from .dataset import PARAMS, make_dataset, sample_dataset
from .performance import error


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


def time(approx_types, datasets):
    n = 1
    runs = 5
    times = {}
    for approx_type in tqdm(approx_types):
        print(approx_type)
        times[approx_type] = np.zeros((len(datasets), runs))
        if approx_type == 'exact':
            continue
        for d, dataset_name in enumerate(datasets):
            dataset, params = make_dataset(dataset_name)
            #nsamples = params[-2]
            X = sample_dataset(10, dataset)
            dim = X.shape[1]
            r = radius(dim, 2 * n)  # 2*n since we use RBF kernel
            b_params = butterfly_params(dim, 2 * n)
            kwargs = {'r': r, 'b_params': b_params}
            for r in range(runs):
                _, _, elapsed = mapping(X, n, 'RBF', approx_type, **kwargs)
                times[approx_type][d, r] = elapsed

    return times
