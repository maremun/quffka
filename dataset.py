#!/usr/bin/env python3
#   encoding: utf-8
#   dataset.py

import numpy as np


def pack_dataset(data, labels, n):
    xtest = data[n:]
    ytest = labels[n:]
    xtrain = data[:n]
    ytrain = labels[:n]
    d = xtrain.shape[1]
    return [xtrain, ytrain, xtest, ytest], d


def get_dataset(name, n):
    path = 'datasets/%s/' % name
    data = np.load(path+'data.npy')
    labels = np.load(path+'labels.npy')
    return pack_dataset(data, labels, n)


def sample_dataset(nsamples, dataset):
    xtrain = dataset[0]
    N = xtrain.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    pick = np.random.choice(idx, nsamples, False)
    sampled = xtrain[pick, :]
    return sampled
