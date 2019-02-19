#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
file  : LISTA_cpss_conv.py
author: Xiaohan Chen
email : chernxh@tamu.edu
date  : 2019-02-17
"""

import numpy as np
import tensorflow as tf

from utils.train import get_train_op

def shrink (input_, theta_):
    return tf.sign (input_) * tf.maximum (tf.abs (input_) - theta_, 0.0)


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
            abs_, 100.0-q, axis=(1,2,3), keep_dims=True)
    # indices of entries with big magnitudes, to which shrinkage should
    # not be applied to
    index_ = tf.logical_and (abs_ > tau_, abs_ > thres_)
    index_ = tf.to_float( index_ )
    # stop gradient at index_, considering it as constant
    index_ = tf.stop_gradient( index_ )
    cindex_ = 1.0 - index_ # complementary index

    return (tf.multiply( index_, inputs_ ) +
            shrink (tf.multiply( cindex_, inputs_ ), tau_ ))


class LISTA_conv_cpss (object):

    """Docstring for LISTA_conv_cp. """

    def __init__(self, nlayer, filters, lam, alpha, percent, max_percent,
                 untied, coord, scope):
        """TODO: to be defined1.

        :nlayer: TODO
        :filters: TODO
        :lam: TODO
        :alpha: TODO
        :untied: TODO
        :coord: TODO
        :scope: TODO

        """
        self._nlayer = nlayer
        self._fh     = filters.shape [0]
        self._fw     = filters.shape [1]
        self._fn     = filters.shape [2]
        self._lam    = lam
        self._alpha  = alpha
        self._p      = percent
        self._mp     = max_percent
        self._untied = untied
        self._coord  = coord
        self._scope  = scope

        # default threshld
        self._theta  = self._lam * self._alpha

        # get the transpose of filters
        tfilters = self._alpha * np.rot90 (filters, k=2, axes=(0,1))
        # set the _fs and _ft by adding new axes to them
        # _fs convs feature maps into image: (fh, fw, fn, 1)
        # _ft convs residual (one image) into feature maps: (fh, fw, 1, fn)
        self._fs =  filters.reshape ((self._fh, self._fw, self._fn, 1))
        self._ft = tfilters.reshape ((self._fh, self._fw, 1, self._fn))

        # percentage of support selection
        ps = np.arange (1, self._nlayer+1) * self._p
        self._ps = np.clip (ps, 0.0, self._mp)

        # set up layers
        self.setup_layers ()


    def setup_layers (self):
        # lists that hold parameters in the network
        self._Ws_     = []
        self._thetas_ = []
        with tf.variable_scope (self._scope, reuse=False) as vs:
            # tf constant for filters
            self._fs_const_ = tf.constant (value=self._fs,
                                           dtype=tf.float32, name='fs')

            self._ft_const_ = tf.constant (value=self._ft,
                                           dtype=tf.float32, name='ft')

            if self._untied == False:
                # tied model
                self._Ws_.append (tf.get_variable (name='W', dtype=tf.float32,
                                                   initializer=self._ft_const_))
                self._Ws_ = self._Ws_ * nlayer

            for i in range (self._nlayer):
                self._thetas_.append (tf.get_variable (name='theta_'+str(i+1),
                                                       dtype=tf.float32,
                                                       initializer=self._theta))
                if self._untied == True:
                    # untied model
                    self._Ws_.append (tf.get_variable (name='W_'+str(i+1),
                                                       dtype=tf.float32,
                                                       initializer=self._ft_const_))


    def inference(self, input_, init_feature_=None, return_recon=False):
        """TODO: Docstring for inference.

        :input_: Batch of images of size (batch_size, h, w, channel=1).
        :init_feature_: Batch of feature maps to be updated of size
                        (batch_size, h+fh-1, w+fw-1, channel=self._fn).
                        None means starting from all zero feature maps.
        :return_recon : Flag that indicates whether to return the reconstructed
                        images.
        :returns: TODO

        """
        # list of features estimated in each layer
        features_ = []
        if return_recon == True:
            recons_ = []

        with tf.variable_scope (self._scope, reuse=True) as vs:
            # set paddding const for residual padding
            ph, pw = self._fh - 1, self._fw - 1
            paddings_ = tf.constant ([[0, 0], [ph, ph], [pw, pw], [0,0]])
                # NOTE: the [0, 0] padding here is for the batch_size axis

            if init_feature_ is None:
                batch_size = tf.shape (input_) [0]
                _, h, w, _  = input_.shape.as_list ()
                feature_ = tf.zeros (shape=(batch_size,
                                            h + self._fh - 1,
                                            w + self._fw - 1,
                                            self._fn),
                                     dtype=tf.float32, name='x_0')
            else:
                feature_ = init_feature_
            features_.append (feature_)

            for t in range (self._nlayer):
                # conv layer to get the reconstructed image
                conv_ = tf.nn.conv2d (input=feature_,
                                      filter=self._fs_const_,
                                      strides=(1,1,1,1),
                                      padding='VALID',
                                      use_cudnn_on_gpu=True,
                                      data_format='NHWC',
                                      name='conv_%d' % (t+1))
                if return_recon:
                    recons_.append (conv_)

                residual_ = input_ - conv_
                # residual padding from (bs, h, w, 1) to
                #                       (bs, h+2fh-2, w+2fw-2, 1)
                padded_res_ = tf.pad (residual_, paddings_, "REFLECT")

                # deconv to calcualte the gradients w.r.t. feature maps
                W_ = self._Ws_ [t]
                grad_ = tf.nn.conv2d (input=padded_res_,
                                      filter=W_,
                                      strides=(1,1,1,1),
                                      padding="VALID",
                                      use_cudnn_on_gpu=True,
                                      data_format='NHWC',
                                      name='deconv_%d' % (t+1))

                # feature_ update
                feature_ = feature_ + grad_

                # thresholding
                theta_ = self._thetas_ [t]
                percent = self._ps [t]
                feature_ = special_shrink (feature_, theta_, percent)
                # append feature_ to feature list
                features_.append (feature_)

            if return_recon:
                conv_ = tf.nn.conv2d (input=feature_,
                                      filter=self._fs_const_,
                                      strides=(1,1,1,1),
                                      padding='VALID',
                                      use_cudnn_on_gpu=True,
                                      data_format='NHWC',
                                      name='conv_%d' % (t+1))
                recons_.append (conv_)
                return features_, recons_
            else:
                return features_


    def setup_training (self, input_, label_, input_val_, label_val_,
                        init_feature_, loss_type, init_lr, decay_rate, lr_decay):
        """TODO: Docstring for setup_training.

        :ih: TODO
        :iw: TODO
        :pnz: TODO
        :SNR: TODO
        :input_: Tensorflow placeholder or tensor. Input of training set.
        :label_: Tensorflow placeholder or tensor. Label for the sparse feature
            maps of training set. If `loss_type` is `recon`, `label_` should be
            `input_` in noiseless reconstruction or noisy image in denoising.
        :input_val_: Tensorflow placeholder or tensor. Input of validation set.
        :label_val_: Tensorflow placeholder or tensor. Label for the sparse
            feature maps of validation set. If `loss_type` is `recon`,
            `label_` should be `input_` in noiseless reconstruction or noisy
            image in denoising.
        :init_feature_: TensorFlow tensor. Initial estimation of feature maps.
        :loss_type: String. 'gt' or 'recon'. Flag that specifies the loss
            function we use, l2 loss w.r.t. the ground truth or the
            reconstruction loss to the input image.
        :init_lr: TODO
        :decay_rate: TODO
        :lr_decay: TODO
        :returns:
            :training_stages: list of training stages

        """
        # infer feature_, feature_val_ from input_, input_val_
        if loss_type == 'gt':
            # predictions are the estimated sparse feature maps
            predicts_ = self.inference (input_, init_feature_, False)
            predicts_val_ = self.inference (input_val_, init_feature_, False)
        elif loss_type == 'recon':
            # predictions are the reconstructions
            _, predicts_ = self.inference (input_, init_feature_, True)
            _, predicts_val_ = self.inference (input_val_, init_feature_, True)
        assert len (predicts_)     == self._nlayer + 1
        assert len (predicts_val_) == self._nlayer + 1
        nmse_denom_     = tf.nn.l2_loss (label_)
        nmse_denom_val_ = tf.nn.l2_loss (label_val_)

        # start setting up training
        training_stages = []

        lrs = [init_lr * decay for decay in lr_decay]

        # setup self.lr_multiplier dictionary
        # learning rate multipliers of each variables
        lr_multiplier = dict()
        for var in tf.trainable_variables():
            lr_multiplier[var.op.name] = 1.0

        # initialize self.train_vars list
        # variables which will be updated in next training stage
        train_vars = []

        for t in range (self._nlayer):
            # layer information for training monitoring
            layer_info = "{scope} T={time}".format (scope=self._scope, time=t+1)

            # set up loss_ and nmse_
            loss_ = tf.nn.l2_loss (predicts_ [t+1] - label_)
            nmse_ = loss_ / nmse_denom_
            loss_val_ = tf.nn.l2_loss (predicts_val_ [t+1] - label_val_)
            nmse_val_ = loss_val_ / nmse_denom_val_

            W_ = self._Ws_ [t]
            theta_ = self._thetas_ [t]

            # train parameters in current layer with initial learning rate
            if W_ not in train_vars:
                var_list = (W_, theta_,)
            else:
                var_list = (theta_,)
            op_ = tf.train.AdamOptimizer(init_lr).minimize (loss_,
                                                            var_list=var_list)
            training_stages.append((layer_info, loss_, nmse_,
                                    loss_val_, nmse_val_, op_, var_list))

            for var in var_list:
                train_vars.append(var)

            # train all variables in current and former layers with decayed
            # learning rate
            for lr in lrs:
                op_ = get_train_op(loss_, train_vars, lr, lr_multiplier)
                training_stages.append ((layer_info + ' lr={}'.format (lr),
                                         loss_,
                                         nmse_,
                                         loss_val_,
                                         nmse_val_,
                                         op_,
                                         tuple (train_vars), ))

            # decay learning rates for trained variables
            for var in train_vars:
                lr_multiplier [var.op.name] *= decay_rate

        return training_stages


