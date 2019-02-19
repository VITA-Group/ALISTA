#-*- coding: utf-8 -*-
"""
ISTA.py
author: chernxh@tamu.edu
date  : 09/26/2017

Python implementation of ISTA algorithm.
"""

import numpy as np
try:
    import cudamat as cm
except ImportError as e:
    print("No module 'cudamat' found. Could not use cudamat.")

def ista(A, y, x0=None, lam=0.01, step=1.0, Tm=10000 ,
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

    scale = 1.001 * np.linalg.norm(A, ord=2)**2
    B   = np.transpose(A) / scale
    tau = lam / scale
    for t in range(int(Tm)):
        z = y - A.dot(xh)
        r = xh + step * np.dot(B, z)
        xh = np.sign(r) * np.maximum( np.abs(r) - tau, 0 )
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


def ista_cm(A , y , x0=None , lam=0.01 , scale=5.0 , Tm=10000 , retpath=False):
    """

    :A: CUDAMatrix, measurement matrix.
    :Y: CUDAMatrix, measurements as column vectors.
    :x0: CUDAMatrix, initial estimation of sparse code.
    :Tm: int, maximum iteration steps.
    :lam: float, threshold lambda in ISTA, should be tuned.
    :returns:
        :Xhat : CUDAMatrix, approximated sparse coding.

    """
    m, n = A.shape
    _, l = y.shape
    if retpath == True :
        xhs = []
    # initial estimation of x, column vectors
    if x0 is None:
        xh = cm.empty((n,l)).assign(0.0)
    else:
        xh = cm.CUDAMatrix(x0)

    z = cm.empty(y.shape)
    r = cm.empty((n, l))
    B = A.transpose()
    B.divide( scale )
    tau = lam / scale
    for t in range(int(Tm)):
        y.subtract( A.dot(xh), target=z )
        xh.add( B.dot(z), target=r )
        cm.soft_threshold( r, tau, target=xh )
        if retpath == True :
            xhs.append (xh.asarray ())

    if retpath == True :
        return xh , np.asarray (xhs)
    else :
        return xh

