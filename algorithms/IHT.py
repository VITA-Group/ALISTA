#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
file          : IHT.py
last_modified : Jul 31, 2018
author        : Xiaohan Chen
email         : chernxh@tamu.edu

This file is an implementation of iterative hard thresholding algorithm.
"""

import numpy as np

def iht(A, y, x0=None, lam=0.01, step=1.0, Tm=500,
        retnmse=False, x=None, retpath=False):
    """

    :A: np array, measurement matrix.
    :Y: np array, measurements as column vectors.
    :x0: np array, initial estimation of sparse code.
    :Tm: int, maximum iteration steps.
    :lam: float, threshold lambda in ISTA, should be tuned.
    :retpath: bool, whether return solution path
    :returns:
        :Xhat : np array, approximated sparse coding.
        :anmse: np array, averaged nmse respect to all samples.

    """
    m, n = A.shape
    _, l = y.shape
    if retnmse == True:
        if x is None:
            raise ValueError ("x should not be None if retnmse is set to True")
        nmse = np.concatenate (([np.ones (1)], np.zeros ((Tm, 1))), axis=0)
        sum_square_X = np.sum (np.square (x.astype (np.float64)))
    if retpath == True:
        xhs = []
    # initial estimation of x, column vectors
    if x0 is None:
        xh = np.zeros((n, l), dtype=np.float64)
    else:
        xh = x0.astype(np.float64)

    B   = np.transpose(A)
    tau = np.sqrt (lam)
    for t in range(int(Tm)):
        r = y - A.dot(xh)
        z = xh + step * np.dot(B, r)
        index = (np.abs (z) >= tau).astype (np.float32)
        xh = index * z
        if retnmse == True:
            nmse [t+1] = np.sum (np.square (xh-x)) / sum_square_X
        if retpath == True:
            xhs.append (xh)

    results = [xh]
    if retnmse == True:
        results.append (nmse)
    if retpath == True:
        results.append (np.asarray (xhs))

    return results
