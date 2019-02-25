#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
file  : prob.py
author: Xiaohan Chen
email : chernxh@tamu.edu
date  : 2019-02-23

Define problem class that is used experiments.
"""

import os
import argparse
import numpy as np
import numpy.linalg as la
from scipy.io import savemat, loadmat

def str2bool(v):
    return v.lower() in ('true', '1')

class Problem(object):

    """
    Problem class is a abstraction of the problem we want to solve: recover
    sparse code x in R^n space from undersampled measurement y = Ax in R^m
    space, where A is a m times n measurement matrix.

    In every problem, we define:
        :A         : numpy array, measurement matrix
        :M, N      : integers, # of rows and cols of matrix A
        :yval, xval: numpy arrays, a set of validation data
        :L         : integer, size of validation data
        :pnz       : hyperparameter about how many non-zero entris in sparse code x
        :SNR       : noise level in measurements
    """

    def __init__(self):
        pass

    def build_prob(self, A, L=1000, pnz=0.1, SNR=40.0, con_num=0.0,
                   col_normalized=True):
        self.A         = A
        # self.A_        = tf.constant( A, dtype=tf.float32, name='A' )
        self.M, self.N = A.shape
        self.L         = L

        self.con_num   = con_num
        self.pnz       = pnz
        self.SNR       = SNR

        self.yval, self.xval \
                = self.gen_samples( self.L )

        if con_num > 0:
            U, _, V = la.svd (A, full_matrices=False)
            s = np.logspace (0, np.log10 (1 / con_num), self.M)
            A = np.dot (U * (s * np.sqrt(self.N) / la.norm(s)), V).astype (np.float32)
        if col_normalized:
            A = A / np.sqrt(np.sum(np.square(A), axis=0, keepdims=True))
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

        print("problem saved to {}".format(path))


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
    return A

def setup_problem (A, L, pnz, SNR, con_num, col_normalized):
    prob = Problem()
    prob.build_prob(A, L, pnz, SNR, con_num, col_normalized)
    return prob

def load_problem( fname ):
    prob = Problem()
    prob.read (fname)
    return prob

parser = argparse.ArgumentParser()
parser.add_argument(
    '-M', '--M', type=int, default=250,
    help="Dimension of measurements.")
parser.add_argument(
    '-N', '--N', type=int, default=500,
    help="Dimension of sparse codes.")
parser.add_argument(
    '-L', '--L', type=int, default=0,
    help="Number of samples for validation (deprecated. please use default).")
parser.add_argument(
    '-P', '--pnz', type=float, default=0.1,
    help="Percent of nonzero entries in sparse codes.")
parser.add_argument(
    '-S', '--SNR', type=str, default='inf',
    help="Strength of noises in measurements.")
parser.add_argument(
    '-C', '--con_num', type=float, default=0.0,
    help="Condition number of measurement matrix. 0 for no modification on condition number.")
parser.add_argument(
    '-CN', '--col_normalized', type=str2bool, default=True,
    help="Flag of whether normalize the columns of the dictionary or sensing matrix.")
parser.add_argument(
    "-lA", "--load_A", type=str, default=None,
    help="Path to the measurement matrix to be loaded.")
parser.add_argument(
    '-ef', '--exp_folder', type=str, default='./experiments',
    help="Root folder for problems and momdels.")
parser.add_argument(
    "-pfn", "--prob_file", type=str, default="prob.npz",
    help="The (base) file name of problem file.")

if __name__ == "__main__":
    config, unparsed = parser.parse_known_args()
    if not config.load_A is None:
        try:
            A = np.load(config.load_A)
            print("matrix loaded from {}. will be used to generate the problem"
                  .format(config.load_A))
        except Exception as e:
            raise ValueError("invalid file {}".format(config.load_A))
        config.M, config.N = A.shape
    else:
        A = np.random.normal(scale=1.0/np.sqrt(config.M),
                             size=(config.M, config.N)).astype(np.float32)
    prob_desc = ('m{}_n{}_k{}_p{}_s{}'.format(
        config.M, config.N, config.con_num, config.pnz, config.SNR))
    prob_folder = os.path.join(config.exp_folder, prob_desc)
    if not os.path.exists(prob_folder):
        os.makedirs(prob_folder)
    out_file = os.path.join(config.exp_folder, prob_desc, config.prob_file)
    if os.path.exists(out_file):
        raise ValueError("specified problem file {} already exists".format(out_file))
    if config.SNR == "inf":
        SNR = np.inf
    else:
        try:
            SNR = float(config.SNR)
        except Exception as e:
            raise ValueError("invalid SNR. use 'inf' or a float number.")
    p = setup_problem(A, config.L, config.pnz, SNR, config.con_num,
                      config.col_normalized)
    p.save(out_file, ftype="npz")

