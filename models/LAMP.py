#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
file  : LAMP.py
author: Xiaohan Chen
email : chernxh@tamu.edu
last_modified: 2018-10-15

Implementation of Learned AMP model.
"""

import numpy as np
import tensorflow as tf
import utils.train

from utils.tf import shrink_lamp
from models.LISTA_base import LISTA_base

class LAMP (LISTA_base):

    """
    Implementation of Learned AMP model.
    """

    def __init__(self, A, T, lam, untied, coord, scope):
        """
        :A      : Instance of Problem class, describing problem settings.
        :T      : Number of layers (depth) of this LISTA model.
        :lam    : Initial value of thresholds of shrinkage functions.
        :untied : Whether weights are shared within layers.
        :coord  :
        :scope  :
        """
        self._A   = A.astype (np.float32)
        self._T   = T
        self._M   = self._A.shape [0]
        self._N   = self._A.shape [1]

        self._lam = lam
        if coord:
            self._lam = np.ones ((self._N, 1), dtype=np.float32) * self._lam

        self._scale = 1.001 * np.linalg.norm (A, ord=2)**2

        self._untied = untied
        self._coord  = coord
        self._scope  = scope

        """ Set up layers."""
        self.setup_layers()


    def setup_layers(self):
        """
        Implementation of LISTA model proposed by LeCun in 2010.

        :prob: Problem setting.
        :T: Number of layers in LISTA.
        :returns:
            :layers: List of tuples ( name, xh_, var_list )
                :name: description of layers.
                :xh: estimation of sparse code at current layer.
                :var_list: list of variables to be trained seperately.

        """
        Bs_   = []
        lams_ = []

        B = (np.transpose (self._A) / self._scale).astype (np.float32)

        with tf.variable_scope (self._scope, reuse=False) as vs:
            # constant
            self._kA_ = tf.constant (value=self._A, dtype=tf.float32)

            if not self._untied: # tied model
                Bs_.append (tf.get_variable (name='B', dtype=tf.float32,
                                             initializer=B))
                Bs_ = Bs_ * self._T

            for t in range (self._T):
                lams_.append (tf.get_variable (name="lam_%d"%(t+1),
                                               dtype=tf.float32,
                                               initializer=self._lam))
                if self._untied: # untied model
                    Bs_.append (tf.get_variable (name='B_%d'%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=B))

        # Collection of all trainable variables in the model layer by layer.
        # We name it as `vars_in_layer` because we will use it in the manner:
        # vars_in_layer [t]
        self.vars_in_layer = list (zip (Bs_, lams_))


    def inference (self, y_, x0_=None, return_recon=False):
        xhs_  = [] # collection of the regressed sparse codes
        if return_recon:
            yhs_  = [] # collection of the reconstructed signals

        if x0_ is None:
            batch_size = tf.shape (y_) [-1]
            xh_ = tf.zeros (shape=(self._N, batch_size), dtype=tf.float32)
        else:
            xh_ = x0_
        xhs_.append (xh_)

        OneOverM = tf.constant (float(1)/self._M, dtype=tf.float32)
        NOverM   = tf.constant (float(self._N)/self._M, dtype=tf.float32)
        vt_ = tf.zeros_like (y_, dtype=tf.float32)

        with tf.variable_scope (self._scope, reuse=True) as vs:
            for t in range (self._T):
                B_, lam_ = self.vars_in_layer [t]

                yh_ = tf.matmul (self._kA_, xh_)
                if return_recon:
                    yhs_.append (yh_)

                xhl0_ = tf.reduce_mean (tf.to_float (tf.abs (xh_)>0), axis=0)
                bt_   = xhl0_ * NOverM

                vt_   = y_ - yh_ + bt_ * vt_
                rvar_ = tf.reduce_sum (tf.square (vt_), axis=0) * OneOverM
                rh_ = xh_ + tf.matmul(B_, vt_)

                xh_ = shrink_lamp (rh_, rvar_, lam_)
                xhs_.append (xh_)

            if return_recon:
                yhs_.append (tf.matmul (self._kA_, xh_))
                return xhs_, yhs_
            else:
                return xhs_

        # B = A.T / (1.001 * la.norm(A,2)**2)
        # B_ =  tf.Variable(B,dtype=tf.float32,name='B_1')
        # By_ = tf.matmul( B_ , self.prob.y_ )

        # lam_     = tf.Variable(self.init_lam, dtype=tf.float32, name='lam_1')
        # rvar_    = tf.reduce_sum(tf.square(self.prob.y_), axis=0) * OneOverM
        # xh_, xhl0_ = eta( By_, rvar_ , lam_ )
        # self.layers.append( ('LAMP T=1', xh_, (B_, lam_,) ) )

        # self.xhs_ = [self.x0_, xh_]

        # vt_ = self.prob.y_
        # for t in range(1, self.T):
        #     bt_   = xhl0_ * NOverM
        #     vt_   = self.prob.y_ - tf.matmul( self.prob.A_ , xh_ ) + bt_ * vt_
        #     rvar_ = tf.reduce_sum(tf.square(vt_), axis=0) * OneOverM
        #     lam_  = tf.Variable(self.init_lam,name='lam_'+str(t+1))

        #     if self.untied:
        #         B_ =  tf.Variable(B, dtype=tf.float32, name='B_'+str(t+1))
        #         var_list = (B_, lam_, )
        #     else:
        #         var_list = (lam_, )

        #     rh_ = xh_ + tf.matmul(B_, vt_)
        #     xh_, xhl0_ = eta( rh_ , rvar_ , lam_ )
        #     self.xhs_.append (xh_)
        #     self.layers.append( ('LAMP T={}'.format(t+1), xh_, var_list ) )

