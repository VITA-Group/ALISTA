#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
file  : prob.py
author: Xiaohan Chen
email : chernxh@tamu.edu
last_modified: 2018-10-03

Define problem class that is used experiments.
"""

import os
import numpy as np
import numpy.linalg as la
# import tensorflow as tf
from scipy.io import savemat, loadmat

class Problem(object):

    """
    Problem class is a abstraction of the problem we want to solve: recover
    sparse code x in R^n space from undersampled measurement y = Ax in R^m
    space, where A is a m times n measurement matrix.

    In every problem, we define:
        :A         : numpy array, measurement matrix
        # :A_        : tensorflow instance of numpy array A
        :M, N      : integers, # of rows and cols of matrix A
        :yval, xval: numpy arrays, a set of validation data
        :L         : integer, size of validation data
        # :y_, x_    : tensorflow placeholders for model training
        :pnz       : hyperparameter about how many non-zero entris in sparse code x
        :SNR       : noise level in measurements
    """

    def __init__(self):
        pass

    def build_prob(self, A, L=1000, pnz=0.1, SNR=40.0, con_num=0.0):
        self.A         = A
        # self.A_        = tf.constant( A, dtype=tf.float32, name='A' )
        self.M, self.N = A.shape
        self.L         = L

        self.con_num   = con_num
        self.pnz       = pnz
        self.SNR       = SNR

        self.yval, self.xval \
                = self.gen_samples( self.L )
        # self.x_ = tf.placeholder( tf.float32, (self.N, None), name='x' )
        # self.y_ = tf.placeholder( tf.float32, (self.M, None), name='y' )

    def measure (self, x, SNR=None):
        """
        Measure sparse code x with matrix A and return the measurement.
        TODO:
          Only consider noiseless setting now.
        """
        if SNR is None:
            SNR = self.SNR
        y   = np.matmul (self.A, x)
        std = np.std (y, axis=0) * np.power (10.0, -SNR/20.0)
        ## The following line is for the compatibility for older versions of
        ## `Numpy` pacakge where the `scale` parameter in `np.randon.normal`
        ## is not allowed to be zero.
        std = np.maximum (std, 10e-50)
        noise = np.random.normal (size=y.shape , scale=std).astype (np.float32)

        return y + noise

    def gen_samples(self, size, pnz=None, SNR=None, probability=None):
        """
        Generate samples (y, x) in current problem setting.
        TODO:
        - About how to generate sparse code x, need to refer to Wotao' paper
          about the strength of signal x.
          Here I just use standard Gaussian.
        """
        if pnz is None:
            pnz = self.pnz

        if SNR is None:
            SNR = self.SNR

        if probability is None:
            probability = pnz
        else:
            assert len (probability) == self.N
            assert np.abs (np.sum (probability) - self.N * pnz) < 1

        bernoulli = np.random.uniform (size=(self.N, size)) <= probability
        bernoulli = bernoulli.astype (np.float32)
        x = bernoulli * np.random.normal (size=(self.N, size)).\
                            astype(np.float32)

        y = self.measure (x, SNR)
        return y, x

    def save(self, path, ftype='npz'):
        """Save current problem settings to npz file or mat file."""
        D = dict(A=self.A,
                 M=self.M,
                 N=self.N,
                 L=self.L,
                 pnz=self.pnz,
                 SNR=self.SNR,
                 con_num=self.con_num,
                 y=self.yval,
                 x=self.xval)

        if path[-4:] != '.' + ftype:
            path = path + '.' + ftype

        if ftype == 'npz':
            np.savez( path, **D )
        elif ftype == 'mat':
            savemat( path, D, oned_as='column' )
        else:
            raise ValueError( 'invalid file type {}'.format( ftype ) )


    def read(self, fname):
        """
        Read saved problem from file.
        """
        if not os.path.exists( fname ):
            raise ValueError('saved problem file {} not found'.format( fname ))
        if fname[-4:] == '.npz':
            # read from saved npz file
            D = np.load( fname )
        elif fname[-4:] == '.mat':
            # read from saved mat file
            D = loadmat( fname )
        else:
            raise ValueError('invalid file type; npz or mat file required')

        if not 'A' in D.keys():
            raise ValueError('invalid input file; matrix A missing')

        for k, v in D.items():
            if k == 'y':
                setattr( self, 'yval' ,v )
            elif k == 'x':
                setattr( self, 'xval', v )
            else:
                setattr( self, k, v )

        self.M, self.N = self.A.shape
        _     , self.L = self.xval.shape

        # # tensorflow part
        # self.x_ = tf.placeholder( tf.float32, (self.N, None), name='x' )
        # self.y_ = tf.placeholder( tf.float32, (self.M, None), name='y' )
        # self.A_ = tf.constant( self.A, dtype=tf.float32, name='A' )

        print( "problem {} successfully loaded".format( fname ) )


def random_A(M, N, con_num=0, col_normalized=True):
    """
    Randomly sample measurement matrix A.
    Curruently I sample A from i.i.d Gaussian distribution with 1./M variance and
    normalize columns.
    TODO: check assumptions on measurement matrix A referring to Wotao Yin's Bregman
    ISS paper.

    :M: integer, dimension of measurement y
    :N: integer, dimension of sparse code x
    :col_normalized:
        boolean, indicating whether normalize columns, default to True
    :returns:
        A: numpy array of shape (M, N)

    """
    A = np.random.normal( scale=1.0/np.sqrt(M), size=(M,N) ).astype(np.float32)
    if con_num > 0:
        U, _, V = la.svd (A, full_matrices=False)
        s = np.logspace (0, np.log10 (1 / con_num), M)
        A = np.dot (U * (s * np.sqrt(N) / la.norm(s)), V).astype (np.float32)
    if col_normalized:
        A = A / np.sqrt (np.sum (np.square (A), axis=0, keepdims=True))
    return A

def setup_problem (M, N, L, pnz, SNR, con_num, col_normalized):
    A = random_A  (M, N, con_num, col_normalized)
    prob = Problem ()
    prob.build_prob (A, L, pnz, SNR, con_num)
    return prob

def load_problem( fname ):
    prob = Problem()
    prob.read (fname)
    return prob

