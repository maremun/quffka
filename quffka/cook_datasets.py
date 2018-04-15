#!/usr/bin/env python3
#   encoding: utf-8
#   cook_datasets.py

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


USPS_DIM = 256
LEU_DIM = 7129


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def parse_data(data, d):
    x = np.empty((len(data), d))
    y = np.empty(len(data))

    for i, line in enumerate(data):
        values = line.strip().split()
        y[i] = float(values[0])
        for j, v in enumerate(values[1:]):
            x[i, j] = float(v.split(':')[1])
    return x, y


def main():
    # Powerplant
    data = pd.read_excel('datasets/Powerplant/CCPP/Folds5x2_pp.xlsx')
    x = data.loc[:, ['AT', 'V', 'AP', 'RH']].values
    y = data.PE.values
    x = StandardScaler().fit_transform(x)

    np.save('datasets/Powerplant/data', x)
    np.save('datasets/Powerplant/labels', y)
    print('Powerplant cooked')
    
    # LETTER
    with open('datasets/LETTER/data', 'r') as f:
        lines = f.readlines()

    data = np.empty([len(lines), 16])
    labels = np.empty(len(lines))
    for i,line in enumerate(lines):
        line = line.strip()
        line = line.split(',')
        data[i, :] = np.array(line[1:])
        labels[i] = ord(line[0]) - 65

    np.save('datasets/LETTER/data', data)
    np.save('datasets/LETTER/labels', labels)
    print('LETTER cooked')

    # USPS
    with open('datasets/USPS/data', 'r') as f:
        train = f.readlines()
    with open('datasets/USPS/data_test', 'r') as f:
        test = f.readlines()

    xtrain, ytrain = parse_data(train, USPS_DIM)
    xtest, ytest = parse_data(test, USPS_DIM)
    data = np.vstack((xtrain, xtest))
    labels = np.hstack((ytrain, ytest))
    np.save('datasets/USPS/data', data)
    np.save('datasets/USPS/labels', labels)
    print('USPS cooked')

    # CIFAR100
    test = unpickle('datasets/CIFAR100/cifar-100-python/test')
    test_data = test[b'data']
    test_labels = test[b'fine_labels']

    train = unpickle('datasets/CIFAR100/cifar-100-python/train')
    train_data = train[b'data']
    train_labels = train[b'fine_labels']
    data = np.vstack([train_data, test_data])
    labels = np.hstack([train_labels, test_labels])
    np.save('datasets/CIFAR100/data.npy', data)
    np.save('datasets/CIFAR100/labels.npy', labels)
    print('CIFAR100 cooked')

    # LEUKEMIA
    with open('datasets/LEUKEMIA/leu', 'r') as f:
        train = f.readlines()
    with open('datasets/LEUKEMIA/leu.t', 'r') as f:
        test = f.readlines()

    xtrain, ytrain = parse_data(train, LEU_DIM)
    xtest, ytest = parse_data(test, LEU_DIM)
    data = np.vstack((xtrain, xtest))
    labels = np.hstack((ytrain, ytest))
    np.save('datasets/LEUKEMIA/data', data)
    np.save('datasets/LEUKEMIA/labels', labels)
    print('LEUKEMIA cooked')
    print('Finished!')
