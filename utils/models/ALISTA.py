#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
file  : ALISTA.py
author: Xiaohan Chen
email : chernxh@tamu.edu
date  : 2019-02-21

Implementation of ALISTA.
"""

import numpy as np
import tensorflow as tf

from utils.tf import shrink_ss, is_tensor
from models.LISTA_base import LISTA_base


class ALISTA(LISTA_base):

    """
    Implementation of deep neural network model.
    """

    def __init__(self, A, T, lam, W, percent, max_percent, coord, scope):
        """
        :prob:     : Instance of Problem class, describing problem settings.
        :T         : Number of layers (depth) of this LISTA model.
        :lam  : Initial value of thresholds of shrinkage functions.
        :untied    : Whether weights are shared within layers.
        """
        self._A    = A.astype(np.float32)
        self._W    = W
        self._T    = T
        self._p    = percent
        self._maxp = max_percent
        self._lam  = lam
        self._M    = self._A.shape[0]
        self._N    = self._A.shape[1]

        self._scale = 1.001 * np.linalg.norm(A, ord=2)**2
        self._theta = (self._lam / self._scale).astype(np.float32)
        if coord:
            self._theta = np.ones((self._N, 1), dtype=np.float32) * self._theta

        self._ps = [(t+1) * self._p for t in range(self._T)]
        self._ps = np.clip(self._ps, 0.0, self._maxp)

        self._coord  = coord
        self._scope  = scope

        """ Set up layers."""
        self.setup_layers()


    def setup_layers(self):
        """ Set up layers of ALISTA.
        """
        alphas_ = [] # step sizes
        thetas_ = [] # thresholds

        with tf.variable_scope(self._scope, reuse=False) as vs:
            # constant
            self._kA_ = tf.constant(value=self._A, dtype=tf.float32)
            if not is_tensor(self._W):
                self._W_ = tf.constant(value=self._W, dtype=tf.float32)
            else:
                self._W_ = self._W
            self._Wt_ = tf.transpose(self._W_, perm=[1,0])

            for t in range(self._T):
                alphas_.append(tf.get_variable(name="alpha_%d"%(t+1),
                                               dtype=tf.float32,
                                               initializer=1.0))
                thetas_.append(tf.get_variable(name="theta_%d"%(t+1),
                                               dtype=tf.float32,
                                               initializer=self._theta))

        # Collection of all trainable variables in the model layer by layer.
        # We name it as `vars_in_layer` because we will use it in the manner:
        # vars_in_layer [t]
        self.vars_in_layer = list(zip(alphas_, thetas_))


    def inference(self, y_, x0_=None):
        xhs_  = [] # collection of the regressed sparse codes

        if x0_ is None:
            batch_size = tf.shape(y_)[-1]
            xh_ = tf.zeros(shape=(self._N, batch_size), dtype=tf.float32)
        else:
            xh_ = x0_
        xhs_.append(xh_)

        with tf.variable_scope(self._scope, reuse=True) as vs:
            for t in range(self._T):
                alpha_, theta_ = self.vars_in_layer[t]
                percent = self._ps[t]

                res_ = y_ - tf.matmul(self._kA_, xh_)
                zh_ = xh_ + alpha_ * tf.matmul(self._Wt_, res_)
                xh_ = shrink_ss(zh_, theta_, percent)
                xhs_.append(xh_)

        return xhs_


