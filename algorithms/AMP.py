#-*- coding: utf-8 -*-
"""
AMP.py
author: chernxh@tamu.edu
date  : 09/26/2017

Python implementation of AMP algorithm.
"""

import numpy as np
from scipy import linalg

def AMP(A, X, Y, X0=None, alf=1.1402, Tm=100):
    """

    :A: np array, measurement matrix
    :X: np array, ground truth sparse coding as column vectors
    :Y: np array, measurements as column vectors
    :Tm: int, maximum iteration steps
    :alf: float, hyperparameter in AMP, should be tuned
    :returns:
        :Xhat : np array, approximated sparse coding
        :anmse: np array, averaged nmse respect to all samples

    """
    m, n = A.shape
    _, l = Y.shape
    if X0 is None:
        Xhat = np.zeros_like(X).astype(np.float64) # initial estimation of X, column vectors
    else:
        Xhat = X0.astype(np.float64)

    nmse = np.zeros((Tm+1, l))
    sum_square_X = np.sum(np.square(X.astype(np.float64)), 0)
    sum_square_X = sum_square_X + ( sum_square_X == 0 )

    supp_gt 	= X != 0
    ratio 		= 0.05
    lsupp_error = []

    Z = np.zeros_like(Y).astype(np.float64)
    for t in range(Tm):
    	# nmse
        nmse[t] = np.sum(np.square(Xhat-X), 0) / sum_square_X
        # supp_err
        Xhat_abs = np.abs(Xhat)
        thres    = np.max(Xhat, axis=0) * ratio
        supp     = Xhat_abs >  thres
        supp_err = np.mean(np.sum(np.logical_xor(supp, supp_gt), axis=0))
        lsupp_error.append(supp_err)

        b = np.sum(np.abs(Xhat) != 0, axis=0) / m
        Z = Y - A.dot(Xhat) + b*Z
        lam = alf / np.sqrt(m) * linalg.norm(Z, ord=2, axis=0)
        R = Xhat + A.T.dot(Z)
        Xhat = np.sign(R) * np.maximum( np.abs(R) - lam, 0 )
        # print("max value in Xhat in %d step is %f" % (t, np.max(Xhat)))

    # nmse
    nmse[Tm] = np.sum(np.square(Xhat-X), 0) / sum_square_X
    # supp_err
    Xhat_abs = np.abs(Xhat)
    thres    = np.max(Xhat, axis=0) * ratio
    supp     = Xhat_abs >  thres
    supp_err = np.mean(np.sum(np.logical_xor(supp, supp_gt), axis=0))
    lsupp_error.append(supp_err)
    return Xhat, np.mean(nmse, 1), np.asarray(lsupp_error)

