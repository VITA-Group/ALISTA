#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bregman.py
author: xhchrn
        chernxh@tamu.edu
last_update: 2017-12-14

Implement naive Bregman iteration algorithm and linearized Bregman ISS
algorithm.
"""

import sys
import numpy as np
from numpy.linalg import norm
try:
    import cudamat as cm
except ImportError as e:
    print("No module 'cudamat' found. Could not use cudamat.")

from algorithms import ISTA

def shrink( z ):
    return np.sign( z ) * np.maximum( np.abs(z) - 1.0, 0.0 )

def linearized_bregman_iss(prob, kappa=64, alpha=20, maxit=1e+4 ):
    """
    Linearized Bregman Iteration Algorithm implementation.

    :prob: TODO
    :kappa: TODO
    :maxit: TODO
    :returns: TODO

    """
    nmse_history = [1.0]
    supp_history = []

    A = prob.A
    M, N = A.shape
    y = prob.yval
    x = prob.xval
    nmse_denom = norm( x, ord=2, axis=0 )

    B = alpha / M * np.transpose( A )
    By = np.matmul( B, y )
    W = (-1.0) / 10. / M * np.matmul( A.T, A )

    z = np.zeros_like( x )
    for t in range( int(maxit) ):
        z = By + z + np.matmul( W, shrink(z) )
        xh = kappa * shrink( z )

        nmse = norm( xh - x, ord=2, axis=0 ) / nmse_denom
        nmse_avg = np.mean( nmse )
        nmse_history.append( nmse_avg )
        supp = np.sum( xh!=0, axis=0 )
        supp_history.append( supp )

    return nmse_history, supp_history

def bregman_iteration(A, y, x, delta, maxout=4, maxit=1e+4):
    """
    Implementation of Bregman iteration algorithm.

    :A: TODO
    :y: TODO
    :x: TODO
    :delta: float, step length in Bregman iteration. A small number.
    :maxout: int, # of outer loops.
    :maxit: int, # of max iterations of ISTA.
    :returns: TODO

    """
    m, n = A.shape
    lam = delta / (2.0 * m)
    xhs = []
    yks = []

    yk = np.zeros_like(y, dtype=np.float64)
    xh = np.zeros_like(x, dtype=np.float64)
    for i in range(int(maxout)):
        yk = yk + (y - np.matmul(A, xh))
        yks.append(yk)

        xh = ISTA.ista( A, yk, lam=lam, Tm=maxit )
        xhs.append( xh )

    return np.asarray(xhs, dtype=np.float64), np.asarray(yks, dtype=np.float64)

def bregman_iteration_cm(A, y, delta, maxout=4, maxit=1e+4):
    """
    Implementation of Bregman iteration algorithm.

    :A: TODO
    :y: TODO
    :delta: float, step length in Bregman iteration. A small number.
    :maxout: int, # of outer loops.
    :maxit: int, # of max iterations of ISTA.
    :returns: TODO

    """
    m, n = A.shape
    _, l = y.shape
    lam = delta / (2.0 * m)
    yks = []
    xhs = []

    scale = 1.001 * np.linalg.norm(A,ord=2)**2
    A_cm = cm.CUDAMatrix(A)
    y_cm = cm.CUDAMatrix(y)
    yk = cm.empty(y.shape).assign(0.0)
    xh = cm.empty((n, l)).assign(0.0)
    for i in range(int(maxout)):
        yk.add(y_cm)
        yk.subtract(A_cm.dot(xh))
        yks.append( yk.asarray() )

        xh = ISTA.ista_cm( A_cm, yk, lam=lam, scale=scale, Tm=maxit )
        xhs.append( xh.asarray() )

    return np.asarray(xhs, dtype=np.float64), np.asarray(yks, dtype=np.float64)

