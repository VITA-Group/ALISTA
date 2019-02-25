#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
file  : utils/tf.py
author: Xiaohan Chen
email : chernxh@tamu.edu
date  : 2019-02-18

Utility functions implemented in TensorFlow, including:
    - miscellaneous
    - shrinkage functions
    - circular padding
    - activations
    - subgradient functions
    - related functions
"""

import tensorflow as tf


############################################################
#######################   Misc   ###########################
############################################################
def is_tensor(x):
    return isinstance(x, (tf.Tensor, tf.SparseTensor, tf.Variable))


############################################################
####################   Shrinkage   #########################
############################################################
def shrink(input_, theta_):
    """
    Soft thresholding function with input input_ and threshold theta_.
    """
    theta_ = tf.maximum( theta_, 0.0 )
    return tf.sign(input_) * tf.maximum( tf.abs(input_) - theta_, 0.0 )


def shrink_free(input_, theta_):
    """
    Soft Shrinkage function without the constraint that the thresholds must be
    greater than zero.
    """
    return tf.sign(input_) * tf.maximum( tf.abs(input_) - theta_, 0.0 )


def shrink_ss(inputs_, theta_, q):
    """
    Special shrink that does not apply soft shrinkage to entries of top q%
    magnitudes.

    :inputs_: TODO
    :thres_: TODO
    :q: TODO
    :returns: TODO

    """
    abs_ = tf.abs(inputs_)
    thres_ = tf.contrib.distributions.percentile(
            abs_, 100.0-q, axis=0, keep_dims=True)

    """
    Entries that are greater than thresholds and in the top q% simultnaneously
    will be selected into the support, and thus will not be sent to the
    shrinkage function.
    """
    index_ = tf.logical_and(abs_ > theta_, abs_ > thres_)
    index_ = tf.to_float(index_)
    """Stop gradient at index_, considering it as constant."""
    index_ = tf.stop_gradient(index_)
    cindex_ = 1.0 - index_ # complementary index

    return (tf.multiply(index_, inputs_) +
            shrink_free(tf.multiply(cindex_, inputs_), theta_ ))


def shrink_ss_batch(inputs_, theta_, q):
    """
    Special shrink that does not apply soft shrinkage to entries of top q%
    magnitudes.

    :inputs_: TODO
    :thres_: TODO
    :q: TODO
    :returns: TODO

    """
    abs_ = tf.abs(inputs_)
    thres_ = tf.contrib.distributions.percentile(abs_, 100.0-q)

    """
    Entries that are greater than thresholds and in the top q% simultnaneously
    will be selected into the support, and thus will not be sent to the
    shrinkage function.
    """
    index_ = tf.logical_and(abs_ > theta_, abs_ > thres_)
    index_ = tf.to_float(index_)
    """Stop gradient at index_, considering it as constant."""
    index_ = tf.stop_gradient(index_)
    cindex_ = 1.0 - index_ # complementary index

    return (tf.multiply(index_, inputs_) +
            shrink_free(tf.multiply(cindex_, inputs_), theta_ ))


def shrink_lamp(r_, rvar_, lam_):
    """
    Implementation of thresholding neuron in Learned AMP model.
    """
    theta_ = tf.maximum(tf.sqrt(rvar_) * lam_, 0.0)
    xh_    = tf.sign(r_) * tf.maximum(tf.abs(r_) - theta_, 0.0)
    return xh_
    # shrink = tf.abs(r_) - theta_
    # xh_    = tf.sign(r_) * tf.maximum(shrink, 0.0)
    # xhl0_  = tf.reduce_mean(tf.to_float(shrink>0), axis=0)
    # return xh_, xhl0_

def hard_shrink(r_, tau_):
    """
    Implementation of hard thresholding neuron.
    """
    return tf.nn.relu(tf.sign(tf.abs(r_) - tau_)) * r_


############################################################
####################    Padding    #########################
############################################################

def circular_pad(input_, filter_, paddings):
    """TODO: Docstring for circular_pad .

    :input_: TODO
    :filter_: TODO
    :paddings: TODO
    :returns: TODO

    """
    pass


############################################################
##################    Subgradient    #######################
############################################################

def subgradient_l1(inputs_, Q_):
    if Q_ is None:
        return tf.sign(inputs_)
    else:
        return tf.sign(inputs_) * Q_

def subgradient_l2(inputs_, Q_):
    if Q_ is None:
        return inputs_
    else:
        return inputs_ * Q_ * Q_

def subgradient_expl1(inputs_, Q_):
    if Q_ is None:
        return tf.sign(inputs_) * tf.exp(tf.abs(inputs_))
    else:
        return tf.sign(inputs_) * tf.exp(tf.abs(inputs_)) * Q_

def subgradient_expl2(inputs_, Q_):
    if Q_ is None:
        return inputs_ * tf.exp(tf.square(inputs_))
    else:
        return inputs_ * tf.exp(tf.square(inputs_)) * Q_

subgradient_funcs = {
        # NOTE: here reweighted l_p norms use the same subgradient function as
        # the normal l_p norms; the difference is only the Q_ parameter
            "l1"   : subgradient_l1,
            "l2"   : subgradient_l2,
            "rel2" : subgradient_l2,
            "expl1": subgradient_expl1,
            "expl2": subgradient_expl2
        }

def get_subgradient_func(norm):
    return subgradient_funcs[norm]



############################################################
##################    Loss Functions   #####################
############################################################

def loss_l1(residual_, Q_):
    if Q_ is None:
        return tf.reduce_sum(tf.abs(residual_))
    else:
        return tf.reduce_sum(tf.abs(residual_) * Q_)

def loss_l2(residual_, Q_):
    if Q_ is None:
        return tf.reduce_sum(tf.square(residual_)) / 2.0
    else:
        return tf.reduce_sum(tf.square(residual_) * Q_ * Q_) / 2.0

def loss_expl1(residual_, Q_):
    if Q_ is None:
        return tf.reduce_sum(tf.exp(tf.abs(residual_)))
    else:
        return tf.reduce_sum(tf.exp(tf.abs(residual_)) * Q_)

def loss_expl2(residual_, Q_):
    if Q_ is None:
        return tf.reduce_sum(tf.exp(tf.square(residual_))) / 2.0
    else:
        return tf.reduce_sum(tf.exp(tf.square(residual_)) * Q_) / 2.0

loss_funcs = {
        # NOTE: here reweighted l_p norms use the same loss function as the
        # normal l_p norms; the difference is only the Q_ parameter
            "l1"     : loss_l1,
            "rel1"   : loss_l1,
            "l2"     : loss_l2,
            "rel2"   : loss_l2,
            "expl1"  : loss_expl1,
            "reexpl1": loss_expl1,
            "expl2"  : loss_expl2,
            "reexpl2": loss_expl2
        }

def get_loss_func(loss, Q):
    return lambda residual: loss_funcs[loss](residual, Q)



############################################################
#####################    Operator    #######################
############################################################

def bmxbm(s, t, batch_first=True):
    """
    Batched matrix and batched matrix multiplication.
    """
    if batch_first:
        equation = "aij,ajk->aik"
    else:
        equation = "ija,jka->ika"

    return tf.einsum(equation, s, t)


def bmxm(s, t, batch_first=True):
    """
    Batched matrix and normal matrix multiplication.
    """
    if batch_first:
        equation = "aij,jk->aik"
    else:
        equation = "ija,jk->ika"

    return tf.einsum(equation, s, t)


def mxbm(s, t, batch_first=True):
    """
    Normal matrix and batched matrix multiplication.
    """
    if batch_first:
        equation = "ij,ajk->aik"
    else:
        equation = "ij,jka->ika"

    return tf.einsum(equation, s, t)

