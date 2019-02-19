#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LISTA_cpss_robust.py
author: xhchrn
        chernxh@tamu.edu
date  : 2018-09-07

Implementation of Learned ISTA with support selection and coupled weights like
in LAMP. With A and W given, we train a set of step sizes and thresholds that
are robust to small turbulence in A.
"""

import numpy as np
import numpy.linalg as la
import tensorflow as tf

import utils.train

# TODO: move shrink functions to utils/shrink.py
def shrink_ft( r_, tau_=1.0 ):
    """
    Implement soft thresholding function with input r_ and threshold tau_.
    """
    # tau_ = tf.maximum( tau_, 0.0 )
    return tf.sign(r_) * tf.maximum( tf.abs(r_) - tau_, 0.0 )


def special_shrink(inputs_, tau_=1.0, q=1.0):
    """
    Special shrink that does not apply soft shrinkage to entries of top q%
    magnitudes.

    :inputs_: TODO
    :thres_: TODO
    :q: TODO
    :returns: TODO

    """
    ### TODO:
    ### multiplying cindex_ to inputs_ could generate different gradients from
    ### to tau_, when tau_ is coordinate-wise thresholds; or not.
    abs_ = tf.abs( inputs_ )
    thres_ = tf.contrib.distributions.percentile(
            abs_, 100.0-q, axis=0, keep_dims=True)
    # indices of entries with big magnitudes, to which shrinkage should
    # not be applied to
    index_ = tf.logical_and (abs_ > tau_, abs_ > thres_)
    index_ = tf.to_float( index_ )
    # stop gradient at index_, considering it as constant
    index_ = tf.stop_gradient( index_ )
    cindex_ = 1.0 - index_ # complementary index

    return tf.multiply (index_, inputs_) +\
           shrink_ft (tf.multiply (cindex_, inputs_), tau_)


class LISTA_cpss_robust (object):

    """
    Implementation of deep neural network model.
    """

    def __init__ (self, m, n, T, theta, percent, max_percent, untied, coord, scope):
        """
        :T          : Integer. Number of layers (depth) of this LISTA model.
        :A          : numpy.ndarray. Measuring matrix/Dictioanry.
        :lam        : Float. Initial value of thresholds of shrinkage functions.
        :percent    : TODO
        :max_percent: TODO
        :untied     : Whether weights are shared within layers.
        :cord       : TODO
        :scope      : TODO
        """
        self._m      = m
        self._n      = n
        self._T      = T
        self._theta  = theta
        self._p      = percent
        self._mp     = max_percent
        self._untied = untied
        self._coord  = coord
        self._scope  = scope

        # theta
        if self._coord:
            self._theta = self._theta * np.ones ([self._n, 1], dtype=np.float32)

        # percentage of support selection
        ps = np.arange (1, self._T+1) * self._p
        self._ps = np.clip (ps, 0.0, self._mp)

        self.setup_layers ()


    def setup_layers (self):
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
        self._step_sizes = []
        self._thetas     = []
        with tf.variable_scope (self._scope, reuse=False) as vs:
            for t in range (self._T):
                ss_    = tf.get_variable ('ss_%d'%(t+1), dtype=tf.float32,
                                          initializer=1.0)
                theta_ = tf.get_variable ('theta_%d'%(t+1), dtype=tf.float32,
                                          initializer=self._theta)
                self._step_sizes.append (ss_)
                self._thetas.append (theta_)


    def inference (self, y_, A_, Wt_, x0_, return_recon):
        xhs_  = [] # collection of the regressed sparse codes
        if return_recon:
            yhs_  = [] # collection of the reconstructed signals
        with tf.variable_scope (self._scope, reuse=True) as vs:
            # init estimation of sparse vectors
            if x0_ is None:
                batch_size = tf.shape (y_) [1]
                xh_ = tf.zeros (shape=(self._n, batch_size), dtype=tf.float32)
            else:
                xh_ = x0_
            xhs_.append (xh_)

            for i in range (self._T):
                ss_ = self._step_sizes [i]
                theta_ = self._thetas [i]
                percent = self._ps [i]

                yh_  = tf.matmul (A_, xh_)
                res_ = y_ - yh_
                zh_  = xh_ + tf.matmul (ss_ * Wt_, res_)
                xh_  = special_shrink (zh_, theta_, percent)

                xhs_.append (xh_)
                if return_recon:
                    yhs_.append  (yh_)

            if return_recon:
                yh_  = tf.matmul (A_, xh_)
                yhs_.append (yh_)
                return xhs_, yhs_
            else:
                return xhs_


    def batch_inference (self, ys_, As_, Wts_, x0_, return_recon):
        """
        Batch inference. Iterate over ys_, As_ and Wts_.
        The first dimension of list_xhs_ stands for the time/iteration in the
        model. list_xhs_ [k] is the stacked outputs of all (y_, A_, Wt_) at the
        step/iteration k.
        """
        list_xhs_ = [[] for i in range (self._T + 1)]
        if return_recon:
            list_yhs_ = [[] for i in range (self._T + 1)]

        # iterate over ys_, As_ and Wts_
        batch_size = ys_.shape.as_list () [0]
        for i in range (batch_size):
            if return_recon:
                xhs_, yhs_ = self.inference (ys_ [i], As_ [i], Wts_ [i],
                                             x0_, return_recon)
                # append yhs_ [t] to list_yhs_ [t] for all t
                for t, yh_ in enumerate (yhs_):
                    list_yhs_ [t].append (yh_)
            else:
                xhs_ = self.inference (ys_ [i], As_ [i], Wts_ [i],
                                       x0_, return_recon)
            # append xhs_ [t] to list_xhs_ [t] for all t
            for t, xh_ in enumerate (xhs_):
                list_xhs_ [t].append (xh_)

        # stacking
        stacked_list_xhs_ = list (map (tf.stack, list_xhs_))
        if return_recon:
            stacked_list_yhs_ = list (map (tf.stack, list_yhs_))

        if return_recon:
            return stacked_list_xhs_, stacked_list_yhs_
        else:
            return stacked_list_xhs_


    # def setup_training(self, init_lr=5e-4, decay_rate=0.5,
    #                    lr_decay=(0.2, 0.02, )):
    #     """
    #     Given prob and layers, we set up training stages according to some hyper-
    #     parameters.

    #     :init_lr: Initial learning rate.
    #     :lr_decay: Learning rate decays.
    #     :returns:
    #         :stages: A list of (name, xh_, loss_, nmse_, op_, var_list).

    #     """
    #     stages = []

    #     lrs = [init_lr * decay for decay in lr_decay]

    #     x_ = self.prob.x_
    #     nmse_denom_ = tf.nn.l2_loss( x_ )

    #     self.decay_rate = decay_rate

    #     # setup self.lr_multiplier dictionary
    #     # learning rate multipliers of each variables
    #     self.lr_multiplier = dict()
    #     for var in tf.trainable_variables():
    #         self.lr_multiplier[var.op.name] = 1.0

    #     # initialize self.train_vars list
    #     # variables which will be updated in next training stage
    #     self.train_vars = []

    #     for l, ( name, xh_, var_list ) in enumerate( self.layers ):
    #         loss_ = tf.nn.l2_loss( xh_ - x_ )
    #         nmse_ = loss_ / nmse_denom_

    #         if var_list is None or len(var_list) == 0:
    #             continue
    #         else:
    #             op_ = tf.train.AdamOptimizer(init_lr)\
    #                     .minimize(loss_, var_list=var_list)
    #             stages.append( ( name, xh_, loss_, nmse_, op_, var_list ) )

    #         for var in var_list:
    #             self.train_vars.append( var )

    #         for lr in lrs:
    #             op_ = self.get_train_op( loss_, self.train_vars, lr )
    #             stages.append((name+' lr='+str(lr),
    #                            xh_,
    #                            loss_,
    #                            nmse_,
    #                            op_,
    #                            tuple(self.train_vars),))

    #         # decay learning rates for trained variables
    #         for var in self.train_vars:
    #             self.lr_multiplier[var.op.name] *= self.decay_rate

    #     self.stages = stages


    # def get_train_op(self, loss_, var_list, lrate):
    #     # get training operator
    #     opt = tf.train.AdamOptimizer( lrate )
    #     grads_vars = opt.compute_gradients( loss_, var_list )
    #     grads_vars_multiplied = []
    #     for grad, var in grads_vars:
    #         grad *= self.lr_multiplier[var.op.name]
    #         grads_vars_multiplied.append( (grad, var) )
    #     return opt.apply_gradients( grads_vars_multiplied )


    # def do_training(self, sess, savefn, batch_size=64, val_step=10,\
    #                 maxit=200000, better_wait=4000 ):
    #     """
    #     Do training actually. Refer to utils/train.py.

    #     :sess       : Tensorflow session,
    #                     in which session we will run the training.
    #     :batch_size : Batch size.
    #     :val_step   : How many steps between two validation.
    #     :maxit      : Max number of iterations in each training stage.
    #     :better_wait: Jump to next stage if no better performance after
    #                     certain # of iterations.

    #     """
    #     self.state = utils.train.do_training(
    #             sess, self.stages, self.prob, savefn, self.scope_name,
    #             batch_size, val_step, maxit, better_wait)


    # def do_cs_training (self, sess,
    #                     train_y_, train_f_, train_x_,
    #                     val_y_  , val_f_  , val_x_,
    #                     savefn, batch_size=64, val_step=10,
    #                     maxit=200000, better_wait=4000, norm_patch=False) :
    #     """
    #     Do training on compressive sensing problem actually. Refer to
    #     utils/train.py.

    #     Param:
    #         :sess    : Tensorflow session.
    #         :trainfn : Path of training data tfrecords.
    #         :valfn   : Path of validation data tfrecords.
    #         :savefn  : Path that trained model to be saved.
    #     Hyperparam:
    #         :batch_size : Batch size.
    #         :val_step   : How many steps between two validation.
    #         :maxit      : Max number of iterations in each training stage.
    #         :better_wait: Jump to next stage if no better performance after
    #                       certain # of iterations.
    #     """
    #     self.state = utils.train.do_cs_training (
    #             sess, self.stages, self.prob,
    #             train_y_, train_f_, train_x_,
    #             val_y_  , val_f_  , val_x_,
    #             savefn, self.scope_name,
    #             batch_size, val_step, maxit, better_wait, norm_patch)


    # def save_trainable_variables( self , sess , savefn ):
    #     """
    #     Save trainable variables in the model to npz file with current value of each
    #     variable in tf.trainable_variables().

    #     :sess: Tensorflow session.
    #     :savefn: File name of saved file.

    #     """
    #     state = getattr ( self , 'state' , {} )
    #     utils.train.save_trainable_variables(
    #             sess , savefn , self.scope_name , **state )


    # def load_trainable_variables( self , sess , savefn ):
    #     """
    #     Load trainable variables from saved file.

    #     :sess: TODO
    #     :savefn: TODO
    #     :returns: TODO

    #     """
    #     self.state = utils.train.load_trainable_variables( sess, savefn )


