#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
file  : TiLISTA.py
author: Xiaohan Chen
email : chernxh@tamu.edu
date  : 2019-02-17

Implementation of TiLISTA --- LISTA with tied weight.
"""

import numpy as np
import tensorflow as tf
import utils.train

from utils.tf import shrink_ss
from models.LISTA_base import LISTA_base


class TiLISTA(LISTA_base):

    """
    Implementation of deep neural network model.
    """

    def __init__(self, A, T, lam, percent, max_percent, coord, scope):
        """
        :prob:     : Instance of Problem class, describing problem settings.
        :T         : Number of layers (depth) of this LISTA model.
        :lam  : Initial value of thresholds of shrinkage functions.
        :untied    : Whether weights are shared within layers.
        """
        self._A = A.astype(np.float32)
        self._T = T
        self._p = percent
        self._maxp = max_percent
        self._lam = lam
        self._M = self._A.shape[0]
        self._N = self._A.shape[1]

        self._scale = 1.001 * np.linalg.norm(A, ord=2)**2
        self._theta = (self._lam / self._scale).astype(np.float32)
        if coord:
            self._theta = np.ones((self._N, 1), dtype=np.float32) * self._theta

        self._ps = [(t+1) * self._p for t in range(self._T)]
        self._ps = np.clip(self._ps, 0.0, self._maxp)

        self._coord = coord
        self._scope = scope

        """ Set up layers."""
        self.setup_layers()


    def setup_layers(self):
        """ Set up layers of ALISTA.
        """
        Ws_ = [] # weight
        alphas_ = [] # step sizes
        thetas_ = [] # thresholds

        W = (np.transpose (self._A) / self._scale).astype (np.float32)

        with tf.variable_scope(self._scope, reuse=False) as vs:
            # constant
            self._kA_ = tf.constant(value=self._A, dtype=tf.float32)
            # tied weight in TiLISTA
            Ws_.append (tf.get_variable (name='W', dtype=tf.float32,
                                         initializer=W))
            Ws_ = Ws_ * self._T

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
        self.vars_in_layer = list(zip(Ws_, alphas_, thetas_))


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
                W_, alpha_, theta_ = self.vars_in_layer[t]
                percent = self._ps[t]

                res_ = y_ - tf.matmul(self._kA_, xh_)
                zh_ = xh_ + alpha_ * tf.matmul(W_, res_)
                xh_ = shrink_ss(zh_, theta_, percent)
                xhs_.append(xh_)

        return xhs_


