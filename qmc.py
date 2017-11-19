#!/usr/bin/env python3
#   encoding: utf-8
#   qmc.py

import ghalton
import numpy as np

from basics import get_d0
from lattice import lattice
from scipy.stats import norm
from sobol import i4_sobol_generate

EPS = 1e-6


def generate_halton_weights(k, n):
    d0 = get_d0(k, n)
    sequencer = ghalton.GeneralizedHalton(n)
    points = np.array(sequencer.get(d0))
    M = norm.ppf(points)
    return M, None


def generate_lattice_weights(k, n):
    d0 = get_d0(k, n)
    z = lattice[:n]
    points = (np.outer(range(d0), z) % d0) / d0 + EPS
    M = norm.ppf(points)
    return M, None


def generate_sobol_weights(k, n):
    d0 = get_d0(k, n)
    points = np.float64(i4_sobol_generate(n, d0, 0)) + EPS
    M = norm.ppf(points).T
    return M, None
