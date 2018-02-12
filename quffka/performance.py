#!/usr/bin/env python3
#   encoding: utf-8
#   performance.py
# TODO documentation

import numpy as np


def error(exact, approx):
    err = np.linalg.norm(exact - approx) / np.linalg.norm(exact)
    return err


def mserror(exact, approx):
    mse = ((exact - approx)**2).mean()
    return mse


def get_estimator_score(data, labels, kernel_type, task):
    '''
    Args:
    =====
    task: if 0 - SVM, 1 - SVR
    kernel_type: any of sklearn's allowed kernel types.
    data: a tuple of matrices; if kernel_type = 'precomputed',
        data[0] is a train set kernel, data[1] - test set kernel,
        otherwise, data[0] is just a train set, data[1] - test set.
    labels: a tuple of target values, labels[0] is train targets,
        labels[1] - test targets.
    '''
    if task == 0:
        estimator = svm.SVC(kernel=kernel_type)
    elif task == 1:
        estimator = svm.SVR(kernel=kernel_type)
    else:
        raise Exception('task=0 for SVM, task=1 for SVR, \
                         other models not implemented')
    estimator.fit(data[0], labels[0])
    predicted = estimator.predict(data[1])
    if task == 0:
        return metrics.accuracy_score(labels[1], predicted)
    else:
        return metrics.r2_score(labels[1], predicted)
