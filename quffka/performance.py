#   encoding: utf-8
#   performance.py
# TODO documentation

import numpy as np

from sklearn.metrics.pairwise import rbf_kernel

from quffka.approximate import kernel
from quffka.basics import generate_random_weights, radius
from quffka.dataset import make_dataset
from quffka.gq import generate_gq_weights
from quffka.mapping import butterfly_params
from quffka.qmc import generate_halton_weights
from quffka.rom import generate_orthogonal_weights, generate_rademacher_weights



def error(exact, approx):
    err = np.linalg.norm(exact - approx) / np.linalg.norm(exact)
    return err


def mserror(exact, approx):
    mse = ((exact - approx)**2).mean()
    return mse


def get_weights(approx_type, d, n):
    if approx_type == 'G':
        M, _ = generate_random_weights(d, n)
    elif approx_type == 'Gort':
        M, _ = generate_random_weights(d, n)
    elif approx_type == 'ROM':
        M, _ = generate_rademacher_weights(d, n)
    elif approx_type == 'QMC':
        M, _ = generate_halton_weights(d, n)
    elif approx_type == 'GQ':
        M, _ = generate_gq_weights(d, n)
    else:
        raise Exception('Sorry, no such approx type :C')
    return M


def get_scores(dataset_name, kernels, approx_type, estimator, N, runs, scores):
    dataset, _ = make_dataset(dataset_name)
    xtrain, ytrain, xtest, ytest = dataset
    d = xtrain.shape[1]
    n1 = xtrain.shape[0]
    n2 = xtest.shape[0]
    r = None
    b_params = None
    M = None

    for k in kernels:
        for n in range(N):
            for i in range(runs):
                if approx_type == 'B':
                    t = 2 * (n+1)
                    r = radius(d, t)
                    b_params = butterfly_params(d, t)
                else:
                    M = get_weights(approx_type, d, n+1)

                precomputed = kernel(xtrain, xtrain, n+1, k, approx_type,
                                     r=r, b_params=b_params, M=M)
                precomputed_test = kernel(xtest, xtrain, n+1, k, approx_type,
                                          r=r, b_params=b_params, M=M)

                clf_pre = estimator(kernel='precomputed')
                clf_pre.fit(precomputed, ytrain)
                scores[dataset_name][k][approx_type][n][i] = clf_pre.score(precomputed_test, ytest)
