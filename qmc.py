#!/usr/bin/env python3
#   encoding: utf-8
#   qmc.py

import ghalton
import numpy as np

from basics import get_D
from lattice import lattice
from scipy.stats import norm
#from sobol import i4_sobol_generate

EPS = 1e-6


def generate_halton_weights(d, n):
    D = get_D(d, n)
    sequencer = ghalton.GeneralizedHalton(d)
    points = np.array(sequencer.get(D))
    M = norm.ppf(points)
    return M, None


def generate_lattice_weights(d, n):
    D = get_D(d, n)
    z = lattice[:d]
    points = (np.outer(range(D), z) % D) / D + EPS
    M = norm.ppf(points)
    return M, None


#def generate_sobol_weights(d, n):
#    D = get_D(d, n)
#    points = np.float64(i4_sobol_generate(d, D, 0)) + EPS
#    M = norm.ppf(points).T
#    return M, None
