#!/usr/bin/env python3
#   encoding: utf-8
#   time.py
# TODO documentation

import ctypes
import pickle as pkl

from experiment import time

approx_types = ['G', 'Gort', 'GQ', 'B']

mkl_rt = ctypes.CDLL('libmkl_rt.so')
print(mkl_rt.mkl_get_max_threads())
mkl_get_max_threads = mkl_rt.mkl_get_max_threads

def mkl_set_num_threads(cores):
    mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))

mkl_set_num_threads(1)
print(mkl_get_max_threads())

times = time(approx_types)
with open('times_%r' % approx_types, 'wb') as f:
    pkl.dump(times, f)
