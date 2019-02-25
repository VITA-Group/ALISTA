#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
file  : ALISTA_robust.py
author: Xiaohan Chen
email : chernxh@tamu.edu
date  : 2019-02-21

Implementation of ALISTA_robust, where the model will take both encoding model A
and weight W as inputs.
"""

import numpy as np
import tensorflow as tf

from utils.tf import shrink_ss, is_tensor
from models.LISTA_base import LISTA_base


class ALISTA_robust(LISTA_base):

    """
    Implementation of deep neural network model.
    """

    def __init__(self, M, N, T, percent, max_percent, coord, scope):
        """
        :prob:     : Instance of Problem class, describing problem settings.
        :T         : Number of layers (depth) of this LISTA model.
        :lam  : Initial value of thresholds of shrinkage functions.
        :untied    : Whether weights are shared within layers.
        """
        self._M    = M
        self._N    = N
        self._T    = T
        self._p    = percent
        self._maxp = max_percent

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

        theta_shape = (self._n, 1) if self._coord else ()

        with tf.variable_scope(self._scope, reuse=False) as vs:
            for t in range(self._T):
                alphas_.append(tf.get_variable(name="alpha_%d"%(t+1),
                                               dtype=tf.float32,
                                               initializer=1.0))
                thetas_.append(tf.get_variable(name="theta_%d"%(t+1),
                                               shape=theta_shape,
                                               dtype=tf.float32))

        # Collection of all trainable variables in the model layer by layer.
        # We name it as `vars_in_layer` because we will use it in the manner:
        # vars_in_layer [t]
        self.vars_in_layer = list(zip(alphas_, thetas_))


    def inference(self, y_, A_, W_, x0_=None):
        assert A_.shape == W_.shape
        if len(A_.shape) > 2:
            return self.batch_inference(y_, A_, W_, x0_=None)

        xhs_  = [] # collection of the regressed sparse codes

        if x0_ is None:
            batch_size = tf.shape(y_)[-1]
            xh_ = tf.zeros(shape=(self._N, batch_size), dtype=tf.float32)
        else:
            xh_ = x0_
        xhs_.append(xh_)

        Wt_ = tf.transpose(W_)
        with tf.variable_scope(self._scope, reuse=True) as vs:
            for t in range(self._T):
                alpha_, theta_ = self.vars_in_layer[t]
                percent = self._ps[t]

                res_ = y_ - tf.matmul(A_, xh_)
                zh_ = xh_ + alpha_ * tf.matmul(Wt_, res_)
                xh_ = shrink_ss(zh_, theta_, percent)
                xhs_.append(xh_)

        return xhs_

    def batch_inference(self, ys_, As_, Ws_, x0_=None):
        """
        Batch inference. Iterate over ys_, As_ and Wts_.
        The first dimension of list_xhs_ stands for the time/iteration in the
        model. list_xhs_ [k] is the stacked outputs of all (y_, A_, Wt_) at the
        step/iteration k.
        """
        # print(ys_.shape)
        # print(As_.shape)
        # print(Ws_.shape)
        list_xhs_ = [[] for i in range(self._T + 1)]

        # iterate over ys_, As_ and Wts_
        batch_size = ys_.shape.as_list()[0]
        for i in range(batch_size):
            xhs_ = self.inference(ys_[i], As_[i], Ws_[i], x0_)
            # append xhs_[t] to list_xhs_[t] for all t
            for t, xh_ in enumerate(xhs_):
                list_xhs_[t].append(xh_)

        # stacking
        stacked_list_xhs_ = list(map(tf.stack, list_xhs_))

        return stacked_list_xhs_


