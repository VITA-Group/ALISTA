#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
file  : AtoW_grad.py
author: Xiaohan Chen
email : chernxh@tamu.edu
date  : 2019-02-20
"""

import numpy as np
import tensorflow as tf
import utils.train

from utils.tf import get_subgradient_func, bmxbm, mxbm

class AtoW_grad(object):

    """Docstring for AtoW_grad. """

    def __init__(self, m, n, T, Binit, eta, loss, Q, scope):
        """TODO: to be defined1.

        :T: TODO
        :loss: TODO

        """
        self._m = m
        self._n = n
        self._Binit = Binit
        self._T = T
        self._loss = loss
        self._eta = eta
        self._Q = Q
        self._scope = scope

        # subgradient function
        self._subgradient_func = get_subgradient_func(loss)

        # setup layers
        self.setup_layers (scope)

    def setup_layers(self, scope):
        """TODO: Docstring for setup_layers.
        :returns: TODO

        """
        with tf.variable_scope (scope, reuse=False) as vs:
            # B initialization
            if isinstance(self._Binit, np.ndarray):
                Binit = (self._eta * self._Binit).astype(np.float32)
                self._Binit_ = tf.constant(value=Binit,
                                           dtype=tf.float32,
                                           name='Binit')
            elif Binit == 'uniform':
                self._Binit_ = tf.random_uniform_initializer(-0.01, 0.01,
                                                             dtype=tf.float32)
            elif Binit == 'normal':
                self._Binit_ = tf.random_normal_initializer(0.0, 0.01,
                                                            dtype=tf.float32)

            # weights
            for i in range (self._T):
                tf.get_variable (name='B_%d'%(i+1),
                                 dtype=tf.float32,
                                 initializer=self._Binit_)

            # Q matrix in loss and subgradient
            if self._Q is None:
                self._Q_ = None
            else:
                self._Q_ = tf.constant (value=self._Q, dtype=tf.float32, name='Q')

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
        Q_ = self._Q_
        with tf.variable_scope (self._scope, reuse=True) as vs:
            for i in range (self._T):
                Z_  = bmxbm (At_, W_, batch_first=True) - self._eye_
                dF_ = self._subgradient_func (Z_, Q_)
                B_  = tf.get_variable ('B_%d'%(i+1))
                W_  = W_ - mxbm (B_, dF_)

        return W_

    def save_trainable_variables (self , sess , savefn):
        """
        Save trainable variables in the model to npz file with current value of each
        variable in tf.trainable_variables().

        :sess: Tensorflow session.
        :savefn: File name of saved file.

        """
        state = getattr (self , 'state' , {})
        utils.train.save_trainable_variables(
                sess, savefn, self._scope, **state )

    def load_trainable_variables (self, sess, savefn):
        """
        Load trainable variables from saved file.

        :sess: TODO
        :savefn: TODO
        :returns: TODO

        """
        self.state = utils.train.load_trainable_variables(sess, savefn)


