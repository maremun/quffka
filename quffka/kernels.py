#!/usr/bin/env python3
#   encoding: utf-8
#   kernels.py
# TODO documentation!
import numpy as np


# ARC-COSINE
def calculate_theta(x, z):
    nx = np.sqrt(np.einsum('ij,ij->i', x, x))
    nz = np.sqrt(np.einsum('ij,ij->i', z, z))
    nxnz = np.einsum('i,j', nx, nz)
    dot = np.einsum('ik,jk->ij', x, z)
    div = np.einsum('ij,ij->ij', dot, 1. / nxnz)
    div[div >= 1.] = 1.
    div[div <= -1.] = -1.
    theta = np.arccos(div)
    return theta, nxnz


def arccos1_kernel(x, z):
    theta, nxnz = calculate_theta(x, z)
    return (nxnz/np.pi)*(np.sin(theta)+(np.pi - theta)*np.cos(theta))


def arccos0_kernel(x, z):
    theta, _ = calculate_theta(x, z)
    return 1. - theta/np.pi


# LINEAR kernel (Gram matrix)
def linear_kernel(x, z):
    K = x.dot(z.T)
    return K


# ANGULAR kernel
def angular_kernel(x, z):
    theta, _ = calculate_theta(x, z)
    return 1. - 2 * theta/np.pi
