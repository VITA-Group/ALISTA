#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LBI.py
author: xhchrn
        chernxh@tamu.edu

Implementation of linearized Bregman iteration algorithm proposed by
Wotao Yin et al. Refer to https://arxiv.org/abs/1104.0262
"""

import numpy as np
from numpy.linalg import norm

def shrink( z ):
    return np.sign( z ) * np.maximum( np.abs(z) - 1.0, 0.0 )

def LBI(prob, kappa=64, alpha=20, maxit=1e+4 ):
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

