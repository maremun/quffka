#!/usr/bin/env python3
#   encoding: utf-8
#   dataset.py

import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


PARAMS = {'USPS': [1, 5, 500, 0, 1, 550, 4500],
          'Powerplant': [1, 5, 500, 0, 1, 550, 8500],
          'LETTER': [1, 5, 500, 0, 1, 550, 10000],
          'MNIST': [1, 5, 50, 0, 1, 550, 50000],
          'CIFAR100': [1, 5, 50, 0, 1, 550, 50000],
          'LEUKEMIA': [1, 5, 50, 0, 1, 550, 38]
          }


def pack_dataset(data, labels, n):
    xtest = data[n:]
    ytest = labels[n:]
    xtrain = data[:n]
    ytrain = labels[:n]
    d = xtrain.shape[1]
    return [xtrain, ytrain, xtest, ytest], d


def load_dataset(name, n):
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


def scale_data(xtrain, xtest):
    xtrain = xtrain.astype('float')
    xtest = xtest.astype('float')
    xtrain /= xtrain.max()
    xtest /= xtest.max()
    return xtrain, xtest


def make_dataset(dataset_name, params=None):
    if params is None:
        params = PARAMS[dataset_name]
    start_deg, max_deg, runs, shift, step, NSAMPLES, delimiter = params

    if dataset_name == 'MNIST':
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
        xtrain = mnist.train.images
        ytrain = mnist.train.labels
        xtest = mnist.test.images
        ytest = mnist.test.labels
        d = xtrain.shape[1]
    else:
        dataset, d = load_dataset(dataset_name, delimiter)
        xtrain, ytrain, xtest, ytest = dataset

    xtrain, xtest = scale_data(xtrain, xtest)
    dataset = [xtrain, ytrain, xtest, ytest]
    X = sample_dataset(NSAMPLES, dataset)
    Y = sample_dataset(NSAMPLES, dataset)
    return X, Y, params
