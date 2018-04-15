#!/usr/bin/env python3
#   encoding: utf-8
#   run.py
# TODO documentation

import ctypes
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import time

from telepyth import TelepythClient

from .approximate import APPROX
from .experiment import relative_errors
from .dataset import make_dataset, sample_dataset


mkl_rt = ctypes.CDLL('libmkl_rt.so')
print(mkl_rt.mkl_get_max_threads())
mkl_get_max_threads = mkl_rt.mkl_get_max_threads
def mkl_set_num_threads(cores):
    mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))
mkl_set_num_threads(30)
print(mkl_get_max_threads())


def main():
    tp = TelepythClient()
    approx_types = ['G', 'Gort']  #, 'ROM', 'QMC', 'GQ', 'B']
    # ghalton cannot generate large enough sequences for datasets CIFAR100 and LEUKEMIA
    approx_types_large_d = ['G', 'Gort', 'ROM', 'GQ', 'B']

    datasets = ['Powerplant', 'LETTER'] #, 'USPS', 'MNIST', 'CIFAR100', 'LEUKEMIA']
    kernels = ['Arccos 1']  # , 'RBF', 'Arccos 0']

    sample_params = None  # [1, 3, 1, 0, 1, 10, 50]
    for name in datasets:
        dataset, params = make_dataset(name, sample_params, 'datasets/')
        start_deg, max_deg, runs, shift, step, nsamples, _ = params
        X = sample_dataset(nsamples, dataset)
        Y = sample_dataset(nsamples, dataset)
        errs = {}
        if X.shape[1] > 784:
            approx_types = approx_types_large_d

        for kernel in kernels:
            print(kernel)
            errs[kernel] = relative_errors(X, Y, kernel, approx_types, start_deg,
                                           max_deg, step, shift, runs)
        directory = 'results/%s/' % name
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(directory + '%s_%r_%r' % (name, kernels, params), 'wb') as f:
            pkl.dump(errs, f)

        print('Finished!')
        tp.send_text('%s dataset done!' % name)
