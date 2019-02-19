#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
file: AtoW_grad.py
author: invert
last_modified: 2018-07-11
"""

import numpy as np
import tensorflow as tf

from utils.tf import get_subgradient_func, bmxbm, mxbm

class AtoW_grad(object):

    """Docstring for AtoW_grad. """

    def __init__(self, m, n, Binit, nlayer, norm, gamma, weight, scope):
        """TODO: to be defined1.

        :nlayer: TODO
        :norm: TODO

        """
        self._m = m
        self._n = n
        self._Binit  = Binit
        self._nlayer = nlayer
        self._norm   = norm
        self._gamma  = gamma
        self._weight = weight
        self._scope  = scope

        # subgradient function
        self._subgradient_func = get_subgradient_func (norm)

        # setup layers
        self.setup_layers (scope)

    def setup_layers(self, scope):
        """TODO: Docstring for setup_layers.
        :returns: TODO

        """
        with tf.variable_scope (scope, reuse=False) as vs:
            # B initialization
            if isinstance (self._Binit, np.ndarray):
                Binit = (self._gamma * self._Binit).astype (np.float32)
                self._Binit_ = tf.constant (value=Binit,
                                            dtype=tf.float32,
                                            name='Binit')
            elif Binit == 'unifrom':
                self._Binit_ = tf.random_uniform_initializer (-0.01, 0.01,
                                                              dtype=tf.float32)
            elif Binit == 'normal':
                self._Binit_ = tf.random_normal_initializer (0.0, 0.01,
                                                             dtype=tf.float32)

            # weights
            for i in range (self._nlayer):
                tf.get_variable (name='B_%d'%(i+1),
                                 dtype=tf.float32,
                                 initializer=self._Binit_)

            # weight matrix in loss and subgradient
            if self._weight is None:
                self._weight_ = None
            else:
                self._weight_ = tf.constant (value=self._weight,
                                             dtype=tf.float32,
                                             name='weight')

            # identity
            eye = np.eye (self._n)
            self._eye_ = tf.constant (value=eye,
                                      dtype=tf.float32,
                                      name='eye')

    def inference(self, A_):
        """TODO: Docstring for function.

        :A_: A tensor or placeholder with shape (batchsize, m, n)
        :returns: TODO

        """
        At_ = tf.transpose (A_, [0,2,1])
        W_ = A_
        weight_ = self._weight_
        with tf.variable_scope (self._scope, reuse=True) as vs:
            for i in range (self._nlayer):
                Z_  = bmxbm (At_, W_, batch_first=True) - self._eye_
                dF_ = self._subgradient_func (Z_, weight_)
                B_  = tf.get_variable ('B_%d'%(i+1))
                W_  = W_ - mxbm (B_, dF_)

        return W_

