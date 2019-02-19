
#-*- coding: utf-8 -*-
"""
FISTA.py
author: chernxh@tamu.edu
date  : 09/27/2017

Python implementation of FISTA algorithm.
"""

import numpy as np

def FISTA(A, X, Y, lam=0.01, Tm=1000):
    """

    :A: np array, measurement matrix
    :X: np array, ground truth sparse coding as column vectors
    :Y: np array, measurements as column vectors
    :Tm: int, maximum iteration steps
    :lam: float, threshold lambda in ISTA, should be tuned
    :returns:
        :Xhat : np array, approximated sparse coding
        :anmse: np array, averaged nmse respect to all samples

    """
    m, n = A.shape
    _, l = Y.shape
    Xhat = np.zeros_like(X).astype(np.float64)
        # initial estimation of X, column vectors
    Xhat_old = np.zeros_like(X).astype(np.float64)
        # intial matrix used to store estimation of X at last step
    nmse = np.concatenate(([np.ones(1)], np.zeros((Tm, 1))), axis=0);
        # nmse(t, k): nmse between Xhat(:, j) and X(:, j) at time t
    sum_square_X = np.sum(np.square(X.astype(np.float64)))
    sum_square_X = sum_square_X + ( sum_square_X == 0 )

    scale = .999 / np.linalg.norm(A.T.dot(A), ord=2)
    B   = scale * np.transpose(A)
    tau = lam * scale
    for t in range(Tm):
        Z = Y - A.dot(Xhat) # residual
        R = Xhat + np.dot(B, Z) + (np.float64(t-2)/(t+1)) * ( Xhat - Xhat_old )
        Xhat_old = Xhat
        Xhat = np.sign(R) * np.maximum( np.abs(R) - tau, 0 )
        nmse[t+1] = np.sum(np.square(Xhat-X)) / sum_square_X

    return Xhat, nmse

