#!/usss;k/bin/env python
# -*- coding: utf-8 -*-

"""
train.py
author: xhchrn
        chernxh@tamu.edu

This file includes codes that set up training, do the training actually.
"""

import sys, os
import tensorflow as tf
import numpy as np

from utils.data import bsd500_cs_inputs


def setup_input_sc(test, p, tbs, vbs, fixval, supp_prob, SNR,
                   magdist, **distargs):
    """TODO: Docstring for function.

    :arg1: TODO
    :returns: TODO

    """
    M, N = p.A.shape

    with tf.name_scope('input'):
        if supp_prob is None:
            supp_prob = p.pnz
        prob_ = tf.constant(value=supp_prob, dtype=tf.float32, name='prob')

        """Sample supports."""
        supp_ = tf.random_uniform(shape=(N, tbs), minval=0.0, maxval=1.0,
                                  name='supp')
        supp_ = tf.to_float(supp_ <= prob_)

        if not test:
            supp_val_ = tf.random_uniform(shape=(N, vbs),
                                          minval=0.0, maxval=1.0,
                                          dtype=tf.float32, name='supp_val')
            supp_val_ = tf.to_float(supp_val_ <= prob_)

        """Sample magnitudes."""
        if magdist == 'normal':
            mag_ = tf.random_normal(shape=(N, tbs),
                                    mean=distargs['mean'],
                                    stddev=distargs['std'],
                                    dtype=tf.float32,
                                    name='mag')
            if not test:
                mag_val_ = tf.random_normal (shape=(N, vbs),
                                             mean=distargs ['mean'],
                                             stddev=distargs ['std'],
                                             dtype=tf.float32,
                                             name='mag')
        elif magdist == 'bernoulli':
            mag_ = (tf.random_uniform(shape=(N, tbs), minval=0.0, maxval=1.0,
                                      dtype=tf.float32)
                    <= distargs['p'])
            mag_ = (tf.to_float (mag_) * distargs ['v0'] +
                    tf.to_float (tf.logical_not (mag_)) * distargs ['v1'])

            if not test:
                mag_val_ = (tf.random_uniform (shape=(N, vbs),
                                               minval=0.0, maxval=1.0,
                                               dtype=tf.float32)
                            <= distargs ['p'])
                mag_val_ = (tf.to_float (mag_val_)
                                * distargs ['v0'] +
                            tf.to_float (tf.logical_not (mag_val_))
                                * distargs ['v1'])


        """Get sparse codes."""
        x_ = supp_ * mag_

        """Measure sparse codes."""
        kA_ = tf.constant (value=p.A, dtype=tf.float32)
        y_  = tf.matmul (kA_, x_)

        """Add noise with SNR."""
        std_ = (tf.sqrt (tf.nn.moments (y_, axes=[0], keep_dims=True) [1])
                    * np.power (10.0, -SNR/20.0))
        noise_ = tf.random_normal (shape=tf.shape (y_), stddev=std_,
                                   dtype=tf.float32, name='noise')
        y_ = y_ + noise_

        if not test and fixval:
            x_val_ = supp_val_ * mag_val_
            y_val_ = tf.matmul (kA_, x_val_)
            std_val_ = (
                tf.sqrt (tf.nn.moments (y_val_, axes=[0], keep_dims=True) [1])
                * np.power (10.0, -SNR/20.0))
            noise_val_ = tf.random_normal (shape=tf.shape (y_val_),
                                           stddev=std_val_, dtype=tf.float32,
                                           name='noise_val')
            y_val_ = y_val_ + noise_val_
            if fixval:
                x_val_ = tf.get_variable (name='label_val', initializer=x_val_)
                y_val_ = tf.get_variable (name='input_val', initializer=y_val_)

        """In the order of `input_, label_, input_val_, label_val_`."""
        if not test:
            return y_, x_, y_val_, x_val_
        else:
            return y_, x_


def setup_input_cs (train_path, val_path, tbs, vbs) :
    y_    , f_     = bsd500_cs_inputs (train_path, tbs, None)
    y_val_, f_val_ = bsd500_cs_inputs (val_path  , vbs, None)

    return y_, f_, y_val_, f_val_


"""*****************************************************************************
Input for curriculum learning of robust training of normal LISTA type networks.
*****************************************************************************"""
def setup_input_robust(A, pmax_sigma, nsigma, pnz, bs_A, bs_x):
    # get the constant defined by A
    A_const_ = tf.constant (value=np.expand_dims (A, -1), dtype=tf.float32)
    m, n, _ = A_const_.shape.as_list ()

    # generate perturbation of A
    perturb_ = tf.random_normal  (shape=(m, n, bs_A))
    psigma_  = tf.random_uniform (shape=(1, 1, bs_A), maxval=pmax_sigma)
    perturb_ = psigma_ * perturb_
    # add perturbation to A and normalize
    # A_perturbed_ has shape (m, n, bs_A)
    A_perturbed_ = A_const_ + perturb_
    colnorms_ = tf.sqrt (tf.reduce_sum (tf.square (A_perturbed_), axis=0, keepdims=True))
    A_perturbed_ = A_perturbed_ / colnorms_

    # generate sparse codes
    # x_ has the shape (n, bs_x)
    bernoulli_ = tf.random_uniform (shape=(n, bs_x), minval=0, maxval=1,
                                    dtype=tf.float32)
    support_   = tf.to_float (bernoulli_ <= pnz)
    magnitude_ = tf.random_normal (shape=(n, bs_x), mean=0, stddev=1.0,
                                   dtype=tf.float32)
    x_ = tf.multiply (support_, magnitude_)

    # measure
    # y_ will first have shape (m, bs_x, bs_A)
    # then we reshape y_ into (m, bs_x * bs_A)
    y_ = tf.einsum ('mna,nx->mxa', A_perturbed_, x_)
    noise_ = tf.random_normal (shape=tf.shape (y_), stddev=nsigma)
    y_ = tf.reshape (y_ + noise_, (m, -1))

    # tile x_ from shape (n, bs_x) to shape (n, bs_x * bs_A)
    x_ = tf.reshape (tf.tile (tf.expand_dims (x_, -1) , [1, 1, bs_A]), (n, -1))

    return A_perturbed_, y_, x_


def setup_sc_training (model, y_, x_, y_val_, x_val_, x0_,
                       init_lr, decay_rate, lr_decay):
    """TODO: Docstring for setup_training.

    :y_: Tensorflow placeholder or tensor.
    :x_: Tensorflow placeholder or tensor.
    :y_val_: Tensorflow placeholder or tensor.
    :x_val_: Tensorflow placeholder or tensor.
    :x0_: TensorFlow tensor. Initial estimation of feature maps.
    :init_lr: TODO
    :decay_rate: TODO
    :lr_decay: TODO
    :returns:
        :training_stages: list of training stages

    """

    """Inference."""
    xhs_     = model.inference (y_    , x0_)
    xhs_val_ = model.inference (y_val_, x0_)

    nmse_denom_     = tf.nn.l2_loss (x_)
    nmse_denom_val_ = tf.nn.l2_loss (x_val_)

    # start setting up training
    training_stages = []

    lrs = [init_lr * decay for decay in lr_decay]

    # setup lr_multiplier dictionary
    # learning rate multipliers of each variables
    lr_multiplier = dict()
    for var in tf.trainable_variables():
        lr_multiplier[var.op.name] = 1.0

    # initialize train_vars list
    # variables which will be updated in next training stage
    train_vars = []

    for t in range (model._T):
        # layer information for training monitoring
        layer_info = "{scope} T={time}".format (scope=model._scope, time=t+1)

        # set up loss_ and nmse_
        loss_ = tf.nn.l2_loss (xhs_ [t+1] - x_)
        nmse_ = loss_ / nmse_denom_
        loss_val_ = tf.nn.l2_loss (xhs_val_ [t+1] - x_val_)
        nmse_val_ = loss_val_ / nmse_denom_val_

        var_list = tuple([var for var in model.vars_in_layer [t]
                               if var not in train_vars])

        # First only train the variables in the `var_list` in current layer.
        op_ = tf.train.AdamOptimizer (init_lr).minimize (loss_,
                                                         var_list=var_list)
        training_stages.append ((layer_info, loss_, nmse_,
                                 loss_val_, nmse_val_, op_, var_list))

        for var in var_list:
            train_vars.append (var)

        # Train all variables in current and former layers with decayed
        # learning rate.
        for lr in lrs:
            op_ = get_train_op (loss_, train_vars, lr, lr_multiplier)
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


def setup_cs_training (model, y_, f_, y_val_, f_val_, x0_,
                       init_lr, decay_rate, lr_decay, lasso_lam):
    """TODO: Docstring for setup_training.

    :y_: Tensorflow placeholder or tensor.
    :x_: Tensorflow placeholder or tensor.
    :y_val_: Tensorflow placeholder or tensor.
    :x_val_: Tensorflow placeholder or tensor.
    :x0_: TensorFlow tensor. Initial estimation of feature maps.
    :init_lr: TODO
    :decay_rate: TODO
    :lr_decay: TODO
    :returns:
        :training_stages: list of training stages


    """
    """Inference."""
    xhs_, fhs_ = model.inference (y_, x0_)
    xhs_val_, fhs_val_ = model.inference (y_val_, x0_)

    nmse_denom_     = tf.nn.l2_loss (f_)
    nmse_denom_val_ = tf.nn.l2_loss (f_val_)

    # start setting up training
    training_stages = []

    lrs = [init_lr * decay for decay in lr_decay]

    # setup lr_multiplier dictionary
    # learning rate multipliers of each variables
    lr_multiplier = dict()
    for var in tf.trainable_variables():
        lr_multiplier[var.op.name] = 1.0

    # initialize train_vars list
    # variables which will be updated in next training stage
    train_vars = []

    for t in range (model._T):
        """Layer information for training monitoring."""
        layer_info = "{scope} T={time}".format (scope=model._scope, time=t+1)

        """Set up `loss_` and `nmse_`."""
        # training
        l2_   = tf.nn.l2_loss (fhs_ [t+1] - f_)
        l1_   = tf.reduce_sum (tf.abs (xhs_ [t+1]))
        loss_ = l2_ + lasso_lam * l1_
        nmse_ = l2_ / nmse_denom_
        # validation
        l2_val_   = tf.nn.l2_loss (fhs_val_ [t+1] - f_val_)
        l1_val_   = tf.reduce_sum (tf.abs (xhs_val_ [t+1]))
        loss_val_ = l2_val_ + lasso_lam * l1_val_
        nmse_val_ = l2_val_ / nmse_denom_val_

        var_list = tuple([var for var in model.vars_in_layer [t]
                               if var not in train_vars])

        # First only train the variables in the `var_list` in current layer.
        op_ = tf.train.AdamOptimizer (init_lr).minimize (loss_,
                                                         var_list=var_list)
        training_stages.append ((layer_info, loss_, nmse_,
                                 loss_val_, nmse_val_, op_, var_list))

        for var in var_list:
            train_vars.append (var)

        # Train all variables in current and former layers with decayed
        # learning rate.
        for lr in lrs:
            op_ = get_train_op (loss_, train_vars, lr, lr_multiplier)
            training_stages.append (dict (name=layer_info + ' lr={}'.format (lr),
                                          loss=loss_,
                                          nmse=nmse_,
                                          loss_val=loss_val_,
                                          nmse_val=nmse_val_,
                                          op=op_,
                                          var_list=tuple (train_vars)))

        # decay learning rates for trained variables
        for var in train_vars:
            lr_multiplier [var.op.name] *= decay_rate

    return training_stages


def get_train_op (loss_, var_list, lr, lr_multiplier):
    """
    Get training operater of loss_ with respect to the variables in the
    var_list, using initial learning rate lr mutliplied with
    variable-specific lr_multiplier.

    :loss_: TensorFlow loss function.
    :var_list: Variables that are to be optimized with respect to loss_.
    :lr: Initial learning rate.
    :lr_multiplier: A dict whose keys are variable.op.name, and values are
                    learning rates multipliers for corresponding vairables.

    :returns: The optmization operator that we want.
    """
    # get training operator
    opt = tf.train.AdamOptimizer (lr)
    grads_vars = opt.compute_gradients (loss_, var_list)
    grads_vars_multiplied = []
    for grad, var in grads_vars:
        grad *= lr_multiplier [var.op.name]
        grads_vars_multiplied.append ((grad, var))
    return opt.apply_gradients (grads_vars_multiplied)


def do_training (sess, stages, savefn, scope, val_step, maxit, better_wait):
    """
    Train the model actually.

    :sess: Tensorflow session. Variables should be initialized or loaded from trained
           model in this session.
    :stages: Training stages info. ( name, xh_, loss_, nmse_, op_, var_list ).
    :prob: Problem instance.
    :batch_size: Batch size.
    :val_step: How many steps between two validation.
    :maxit: Max number of iterations in each training stage.
    :better_wait: Jump to next training stage in advance if nmse_ no better after
                  certain number of steps.
    :done: name of stages that has been done.

    """
    if os.path.exists ( savefn ):
        sys.stdout.write ('Pretrained model found. Loading...\n')
        state = load_trainable_variables (sess , savefn)
    else:
        state = {}

    done = state.get ('done' , [])
    log  = state.get ('log' , [])

    # for name, loss_, nmse_, loss_val_, nmse_val_, op_, var_list in stages:
    for name, loss_, nmse_, loss_val_, nmse_val_, op_, var_list in stages:
        """Skip stage done already."""
        if name in done:
            sys.stdout.write ( 'Already did {}. Skipping\n'.format(name))
            continue

        # print stage information
        var_disc = 'fine tuning ' + ','.join([v.name for v in var_list])
        print (name + ' ' + var_disc)

        nmse_hist_val = []
        for i in range (maxit+1):

            _, loss_tr, nmse_tr = sess.run ([op_, loss_, nmse_])
            db_tr = 10. * np.log10(nmse_tr)

            if i % val_step == 0:
                nmse_val, loss_val = sess.run ([nmse_val_, loss_val_])

                if np.isnan (nmse_val):
                    raise RuntimeError ('nmse is nan. exiting...')

                nmse_hist_val = np.append (nmse_hist_val, nmse_val)
                db_best_val = 10. * np.log10 (nmse_hist_val.min())
                db_val = 10. * np.log10 (nmse_val)
                sys.stdout.write(
                        "\r| i={i:<7d} | loss_tr={loss_tr:.6f} | "
                        "db_tr={db_tr:.6f}dB | loss_val ={loss_val:.6f} | "
                        "db_val={db_val:.6f}dB | (best={db_best_val:.6f})"\
                            .format(i=i, loss_tr=loss_tr, db_tr=db_tr,
                                    loss_val=loss_val, db_val=db_val,
                                    db_best_val=db_best_val))
                sys.stdout.flush()
                if i % (100 * val_step) == 0:
                    print('')
                    age_of_best = (len(nmse_hist_val) -
                                   nmse_hist_val.argmin() - 1)
                    # If nmse has not improved for a long time, jump to the
                    # next training stage.
                    if age_of_best * val_step > better_wait:
                        break

        done = np.append (done , name)
        # TODO: add log

        state [ 'done' ] = done
        state [ 'log' ] = log

        save_trainable_variables (sess ,savefn , scope, **state)


def do_cs_training (sess, stages, prob,
                    train_y_, train_f_, train_x_,
                    val_y_  , val_f_  , val_x_,
                    savefn, scope, batch_size=64,
                    val_step=10, maxit=200000, better_wait=4000,
                    norm_patch=False):
    """
    Train the model actually.

    Params:
        :sess: Tensorflow session. Variables should be initialized or loaded
               from package import modulepackage import modulepackage import
               moduletrained model in this session.
        :stages: Training stages info.
                 ( name, xh_, loss_, nmse_, op_, var_list ).
        :prob: Problem instance.
        :train_x_: Data loader of x for training data.
        :train_y_: Data loader of y for training data.
        :  val_x_: Data loader of x for validation data.
        :  val_y_: Data loader of y for validation data.
    Hyperparams:
        :batch_size: Batch size.
        :val_step: How many steps between two validation.
        :maxit: Max number of iterations in each training stage.
        :better_wait: Jump to next training stage in advance if nmse_ no better
                      after certain number of steps.
    Returns:
        :state: training states that are logged.
            :done: name of stages that has been done.
            :log : TODO

    """

    scope = tf.get_variable_scope()
    scope.reuse_variables()

    # load pretrained model and state
    if os.path.exists ( savefn ):
        sys.stdout.write ( 'Pretrained model found. Loading...\n' )
        state = load_trainable_variables( sess , savefn )
    else:
        state = {}
    done = state.get ( 'done' , [] )
    log  = state.get ( 'log' , [] )

    scale = 255.0 if norm_patch else 1.0

    val_x, val_y = sess.run ([val_x_, val_y_])
    val_x /= scale
    val_y /= scale
    for name, xh_, loss_, nmse_, op_, var_list in stages:
        if name in done:
            sys.stdout.write ( 'Already did {}. Skipping\n'.format( name ) )
            continue
        if len(var_list) > 0:
            var_disc = 'fine tuning ' + ','.join( [v.name for v in var_list] )
        else:
            var_disc = 'fine tuning all ' + ','.join( [v.name for v in tf.trainable_variables()] )

        print( name + ' ' + var_disc )
        val_nmse_history = []
        for i in range( maxit+1 ):

            train_x, train_y = sess.run ([train_x_, train_y_])
            train_x /= scale
            train_y /= scale
            _, train_loss, train_nmse = \
                sess.run([op_, loss_, nmse_],
                         feed_dict={prob.y_:train_y, prob.x_:train_x})
            train_nmse_dB = 10. * np.log10( train_nmse )

            if i % val_step == 0:
                val_nmse, val_loss, xh = sess.run(
                        [nmse_, loss_, xh_],
                        feed_dict={prob.y_:val_y, prob.x_:val_x}
                        )
                if np.isnan( val_nmse ):
                    raise RuntimeError( 'nmse is nan. exiting...' )

                val_nmse_history = np.append( val_nmse_history, val_nmse )
                val_nmse_best_dB = 10. * np.log10( val_nmse_history.min() )
                val_nmse_dB = 10. * np.log10( val_nmse )
                sys.stdout.write(
                        "\ri={i:<7d} train_loss={train_loss:.6f} "
                        "train_nmse={train_nmse:.6f} dB val_loss ={val_loss:.6f} "
                        "val_nmse  ={val_nmse:.6f} dB (best={best:.6f}) "
                        "magnitude ={magnitude:.6f}"\
                            .format(i=i,
                            train_loss=train_loss, train_nmse=train_nmse_dB,
                            val_loss=val_loss, val_nmse=val_nmse_dB,
                            best=val_nmse_best_dB,
                            magnitude=np.max (np.abs (xh)))
                        )
                sys.stdout.flush()
                if i % (100 * val_step) == 0:
                    print('')
                    age_of_best = len(val_nmse_history) - val_nmse_history.argmin() - 1
                    # if nmse has not improved for a long time, jump to the next training stage
                    if age_of_best * val_step > better_wait:
                        break

        done = np.append ( done , name )
        # TODO: add log

        state [ 'done' ] = done
        state [ 'log' ] = log

        save_trainable_variables ( sess , savefn , scope , **state )

    return state


def do_csrecon_training (sess, stages, prob, f_,
                         train_y_, train_f_, train_x_,
                         val_y_  , val_f_  , val_x_,
                         savefn, scope, batch_size=64,
                         val_step=10, maxit=200000, better_wait=4000,
                         norm_patch=False):
    """
    Train the model actually.

    Params:
        :sess: Tensorflow session. Variables should be initialized or loaded
               from package import modulepackage import modulepackage import
               moduletrained model in this session.
        :stages: Training stages info.
                 ( name, xh_, loss_, nmse_, op_, var_list ).
        :prob: Problem instance.
        :train_x_: Data loader of x for training data.
        :train_y_: Data loader of y for training data.
        :  val_x_: Data loader of x for validation data.
        :  val_y_: Data loader of y for validation data.
    Hyperparams:
        :batch_size: Batch size.
        :val_step: How many steps between two validation.
        :maxit: Max number of iterations in each training stage.
        :better_wait: Jump to next training stage in advance if nmse_ no better
                      after certain number of steps.
    Returns:
        :state: training states that are logged.
            :done: name of stages that has been done.
            :log : TODO

    """

    scope = tf.get_variable_scope()
    scope.reuse_variables()

    # load pretrained model and state
    if os.path.exists ( savefn ):
        sys.stdout.write ( 'Pretrained model found. Loading...\n' )
        state = load_trainable_variables( sess , savefn )
    else:
        state = {}
    done = state.get ( 'done' , [] )
    log  = state.get ( 'log' , [] )

    scale = 255.0 if norm_patch else 1.0

    val_y, val_f = sess.run ([val_y_, val_f_])
    val_y /= scale
    val_f /= scale
    for name, xh_, loss_, nmse_, op_, var_list in stages:
        if name in done:
            sys.stdout.write ( 'Already did {}. Skipping\n'.format( name ) )
            continue
        if len(var_list) > 0:
            var_disc = 'fine tuning ' + ','.join( [v.name for v in var_list] )
        else:
            var_disc = 'fine tuning all ' + ','.join( [v.name for v in tf.trainable_variables()] )

        print( name + ' ' + var_disc )
        val_nmse_history = []
        for i in range( maxit+1 ):

            train_y, train_f = sess.run ([train_y_, train_f_])
            train_y /= scale
            train_f /= scale
            _, train_loss, train_nmse = \
                sess.run([op_, loss_, nmse_],
                         feed_dict={prob.y_:train_y, f_:train_f})
            train_nmse_dB = 10. * np.log10( train_nmse )

            if i % val_step == 0:
                val_nmse, val_loss = sess.run(
                        [nmse_, loss_,],
                        feed_dict={prob.y_:val_y, f_:val_f}
                        )
                if np.isnan( val_nmse ):
                    raise RuntimeError( 'nmse is nan. exiting...' )

                val_nmse_history = np.append( val_nmse_history, val_nmse )
                val_nmse_best_dB = 10. * np.log10( val_nmse_history.min() )
                val_nmse_dB = 10. * np.log10( val_nmse )
                sys.stdout.write(
                        "\ri={i:<7d} train_loss={train_loss:.6f} "
                        "train_nmse={train_nmse:.6f} dB val_loss ={val_loss:.6f} "
                        "val_nmse  ={val_nmse:.6f} dB (best={best:.6f})"\
                            .format(i=i,
                            train_loss=train_loss, train_nmse=train_nmse_dB,
                            val_loss=val_loss, val_nmse=val_nmse_dB,
                            best=val_nmse_best_dB)
                        )
                sys.stdout.flush()
                if i % (100 * val_step) == 0:
                    print('')
                    age_of_best = len(val_nmse_history) - val_nmse_history.argmin() - 1
                    # if nmse has not improved for a long time, jump to the next training stage
                    if age_of_best * val_step > better_wait:
                        break

        done = np.append ( done , name )
        # TODO: add log

        state [ 'done' ] = done
        state [ 'log' ] = log

        save_trainable_variables ( sess , savefn , scope , **state )

    return state


# def do_csrecon_dropout_training (sess, stages, prob, f_,
#                                  keep_prob_, keep_prob,
#                                  train_y_, train_f_, train_x_,
#                                  val_y_  , val_f_  , val_x_,
#                                  savefn, scope, batch_size=64,
#                                  val_step=10, maxit=200000, better_wait=4000,
#                                  norm_patch=False):
#     """
#     Train the model actually.
#
#     Params:
#         :sess: Tensorflow session. Variables should be initialized or loaded
#                from package import modulepackage import modulepackage import
#                moduletrained model in this session.
#         :stages: Training stages info.
#                  ( name, xh_, loss_, nmse_, op_, var_list ).
#         :prob: Problem instance.
#         :train_x_: Data loader of x for training data.
#         :train_y_: Data loader of y for training data.
#         :  val_x_: Data loader of x for validation data.
#         :  val_y_: Data loader of y for validation data.
#     Hyperparams:
#         :batch_size: Batch size.
#         :val_step: How many steps between two validation.
#         :maxit: Max number of iterations in each training stage.
#         :better_wait: Jump to next training stage in advance if nmse_ no better
#                       after certain number of steps.
#     Returns:
#         :state: training states that are logged.
#             :done: name of stages that has been done.
#             :log : TODO
#
#     """
#
#     scope = tf.get_variable_scope()
#     scope.reuse_variables()
#
#     # load pretrained model and state
#     if os.path.exists ( savefn ):
#         sys.stdout.write ( 'Pretrained model found. Loading...\n' )
#         state = load_trainable_variables( sess , savefn )
#     else:
#         state = {}
#     done = state.get ( 'done' , [] )
#     log  = state.get ( 'log' , [] )
#
#     scale = 255.0 if norm_patch else 1.0
#
#     val_y, val_f = sess.run ([val_y_, val_f_])
#     val_y /= scale
#     val_f /= scale
#     for name, xh_, loss_, nmse_, op_, var_list in stages:
#         if name in done:
#             sys.stdout.write ( 'Already did {}. Skipping\n'.format( name ) )
#             continue
#         if len(var_list) > 0:
#             var_disc = 'fine tuning ' + ','.join( [v.name for v in var_list] )
#         else:
#             var_disc = 'fine tuning all ' + ','.join( [v.name for v in tf.trainable_variables()] )
#
#         print( name + ' ' + var_disc )
#         val_nmse_history = []
#         for i in range( maxit+1 ):
#
#             train_y, train_f = sess.run ([train_y_, train_f_])
#             train_y /= scale
#             train_f /= scale
#             _, train_loss, train_nmse = \
#                 sess.run([op_, loss_, nmse_],
#                          feed_dict={
#                             prob.y_   :train_y,
#                             f_        :train_f,
#                             keep_prob_: keep_prob
#                         })
#             train_nmse_dB = 10. * np.log10( train_nmse )
#
#             if i % val_step == 0:
#                 val_nmse, val_loss = sess.run(
#                         [nmse_, loss_,],
#                         feed_dict={prob.y_:val_y, f_:val_f}
#                         )
#                 if np.isnan( val_nmse ):
#                     raise RuntimeError( 'nmse is nan. exiting...' )
#
#                 val_nmse_history = np.append( val_nmse_history, val_nmse )
#                 val_nmse_best_dB = 10. * np.log10( val_nmse_history.min() )
#                 val_nmse_dB = 10. * np.log10( val_nmse )
#                 sys.stdout.write(
#                         "\ri={i:<7d} train_loss={train_loss:.6f} "
#                         "train_nmse={train_nmse:.6f} dB val_loss ={val_loss:.6f} "
#                         "val_nmse  ={val_nmse:.6f} dB (best={best:.6f})"\
#                             .format(i=i,
#                             train_loss=train_loss, train_nmse=train_nmse_dB,
#                             val_loss=val_loss, val_nmse=val_nmse_dB,
#                             best=val_nmse_best_dB)
#                         )
#                 sys.stdout.flush()
#                 if i % (100 * val_step) == 0:
#                     print('')
#                     age_of_best = len(val_nmse_history) - val_nmse_history.argmin() - 1
#                     # if nmse has not improved for a long time, jump to the next training stage
#                     if age_of_best * val_step > better_wait:
#                         break
#
#         done = np.append ( done , name )
#         # TODO: add log
#
#         state [ 'done' ] = done
#         state [ 'log' ] = log
#
#         save_trainable_variables ( sess , savefn , scope , **state )
#
#     return state


def do_csjoint_training (sess, stages, prob, f_,
                         train_y_, train_f_, train_x_,
                         val_y_  , val_f_  , val_x_,
                         savefn, scope, batch_size=64,
                         val_step=10, maxit=200000, better_wait=4000,
                         norm_patch=False):
    """
    Train the model actually.

    Params:
        :sess: Tensorflow session. Variables should be initialized or loaded
               from package import modulepackage import modulepackage import
               moduletrained model in this session.
        :stages: Training stages info.
                 ( name, xh_, loss_, nmse_, op_, var_list ).
        :prob: Problem instance.
        :train_x_: Data loader of x for training data.
        :train_y_: Data loader of y for training data.
        :  val_x_: Data loader of x for validation data.
        :  val_y_: Data loader of y for validation data.
    Hyperparams:
        :batch_size: Batch size.
        :val_step: How many steps between two validation.
        :maxit: Max number of iterations in each training stage.
        :better_wait: Jump to next training stage in advance if nmse_ no better
                      after certain number of steps.
    Returns:
        :state: training states that are logged.
            :done: name of stages that has been done.
            :log : TODO

    """

    scope = tf.get_variable_scope()
    scope.reuse_variables()

    # load pretrained model and state
    if os.path.exists ( savefn ):
        sys.stdout.write ( 'Pretrained model found. Loading...\n' )
        state = load_trainable_variables( sess , savefn )
    else:
        state = {}
    done = state.get ( 'done' , [] )
    log  = state.get ( 'log' , [] )

    scale = 255.0 if norm_patch else 1.0

    val_x, val_y, val_f = sess.run ([val_x_, val_y_, val_f_])
    val_x /= scale
    val_y /= scale
    val_f /= scale
    for name, xh_, loss_, nmse_, op_, var_list in stages:
        if name in done:
            sys.stdout.write ( 'Already did {}. Skipping\n'.format( name ) )
            continue
        if len(var_list) > 0:
            var_disc = 'fine tuning ' + ','.join( [v.name for v in var_list] )
        else:
            var_disc = 'fine tuning all ' + ','.join( [v.name for v in tf.trainable_variables()] )

        print( name + ' ' + var_disc )
        val_nmse_history = []
        for i in range( maxit+1 ):

            train_x, train_y, train_f = sess.run ([train_x_, train_y_, train_f_])
            train_x /= scale
            train_y /= scale
            train_f /= scale
            _, train_loss, train_nmse = \
                sess.run([op_, loss_, nmse_],
                        feed_dict={prob.y_:train_y, prob.x_:train_x, f_:train_f})
            train_nmse_dB = 10. * np.log10( train_nmse )

            if i % val_step == 0:
                val_nmse, val_loss, xh = sess.run(
                        [nmse_, loss_, xh_],
                        feed_dict={prob.y_:val_y, prob.x_:val_x, f_:val_f}
                        )
                if np.isnan( val_nmse ):
                    raise RuntimeError( 'nmse is nan. exiting...' )

                val_nmse_history = np.append( val_nmse_history, val_nmse )
                val_nmse_best_dB = 10. * np.log10( val_nmse_history.min() )
                val_nmse_dB = 10. * np.log10( val_nmse )
                sys.stdout.write(
                        "\ri={i:<7d} train_loss={train_loss:.6f} "
                        "train_nmse={train_nmse:.6f} dB val_loss ={val_loss:.6f} "
                        "val_nmse  ={val_nmse:.6f} dB (best={best:.6f}) "
                        "magnitude ={magnitude:.6f}"\
                            .format(i=i,
                            train_loss=train_loss, train_nmse=train_nmse_dB,
                            val_loss=val_loss, val_nmse=val_nmse_dB,
                            best=val_nmse_best_dB,
                            magnitude=np.max (np.abs (xh)))
                        )
                sys.stdout.flush()
                if i % (100 * val_step) == 0:
                    print('')
                    age_of_best = len(val_nmse_history) - val_nmse_history.argmin() - 1
                    # if nmse has not improved for a long time, jump to the next training stage
                    if age_of_best * val_step > better_wait:
                        break

        done = np.append ( done , name )
        # TODO: add log

        state [ 'done' ] = done
        state [ 'log' ] = log

        save_trainable_variables ( sess , savefn , scope , **state )

    return state


def save_trainable_variables (sess, filename, scope, **kwargs):
    """
    Save trainable variables in the model to npz file with current value of
    each variable in tf.trainable_variables().

    :sess: Tensorflow session.
    :filename: File name of saved file.
    :scope: Name of the variable scope that we want to save.
    :kwargs: Other arguments that we want to save.

    """
    save = dict ()
    for v in tf.trainable_variables ():
        if scope in v.name:
            save [str (v.name)] = sess.run (v)

    # file name suffix check
    if filename [-4:] != '.npz':
        filename = filename + '.npz'

    save.update (kwargs)
    np.savez (filename , **save)


def load_trainable_variables (sess, filename):
    """
    Load trainable variables from saved file.

    :sess: TODO
    :filename: TODO
    :returns: TODO

    """
    other = dict ()
    # file name suffix check
    if filename [-4:] != '.npz':
        filename = filename + '.npz'
    if not os.path.exists (filename):
        raise ValueError (filename + ' not exists')

    tv = dict ([(str(v.name), v) for v in tf.trainable_variables ()])
    for k, d in np.load (filename).items ():
        if k in tv:
            print ('restoring ' + k)
            sess.run (tf.assign (tv[k], d))
        else:
            other [k] = d

    return other

