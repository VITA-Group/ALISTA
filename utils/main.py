#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
file  : main.py
author: Xiaohan Chen
email : chernxh@tamu.edu
last_modified: 2018-10-13

Main script. Start running model from main.py.
"""

import os , sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUIET!!!!

# timing
import time
from datetime import timedelta

from config import get_config
import utils.prob as problem
import utils.data as data
import utils.train as train

import numpy as np
import tensorflow as tf
try :
    from PIL import Image
    from sklearn.feature_extraction.image \
            import extract_patches_2d, reconstruct_from_patches_2d
except Exception as e :
    pass


def imread_CS_py(im_fn, patch_size, stride):
    im_org = np.array (Image.open (im_fn), dtype='float32')
    H, W   = im_org.shape
    num_rpatch = (H - patch_size + stride - 1) // stride + 1
    num_cpatch = (W - patch_size + stride - 1) // stride + 1
    H_pad = patch_size + (num_rpatch - 1) * stride
    W_pad = patch_size + (num_cpatch - 1) * stride
    im_pad = np.zeros ((H_pad, W_pad), dtype=np.float32)
    im_pad [:H, :W] = im_org

    return im_org, H, W, im_pad, H_pad, W_pad


def img2col_py(im_pad, patch_size, stride):
    [H, W] = im_pad.shape
    num_rpatch = (H - patch_size) / stride + 1
    num_cpatch = (W - patch_size) / stride + 1
    num_patches = int (num_rpatch * num_cpatch)
    img_col = np.zeros ([patch_size**2, num_patches])
    count = 0
    for x in range(0, H-patch_size+1, stride):
        for y in range(0, W-patch_size+1, stride):
            img_col[:, count] = im_pad[x:x+patch_size, y:y+patch_size].reshape([-1])
            count = count + 1
    return img_col


def col2im_CS_py(X_col, patch_size, stride, H, W, H_pad, W_pad):
    X0_rec = np.zeros ((H_pad, W_pad))
    counts = np.zeros ((H_pad, W_pad))
    k = 0
    for x in range(0, H_pad-patch_size+1, stride):
        for y in range(0, W_pad-patch_size+1, stride):
            X0_rec[x:x+patch_size, y:y+patch_size] += X_col[:,k].\
                    reshape([patch_size, patch_size])
            counts[x:x+patch_size, y:y+patch_size] += 1
            k = k + 1
    X0_rec /= counts
    X_rec = X0_rec[:H, :W]
    return X_rec


def setup_model (config , **kwargs) :
    untiedf = 'u' if config.untied else 't'
    coordf  = 'c' if config.coord  else 's'

    if config.net == 'LISTA' :
        """LISTA"""
        config.model = ("LISTA_T{T}_lam{lam}_{untiedf}_{coordf}_{exp_id}"
                        .format (T=config.T, lam=config.lam, untiedf=untiedf,
                                 coordf=coordf, exp_id=config.exp_id))
        from models.LISTA import LISTA
        model = LISTA (kwargs ['A'], T=config.T, lam=config.lam,
                       untied=config.untied, coord=config.coord,
                       scope=config.scope)

    if config.net == 'LAMP' :
        """LAMP"""
        config.model = ("LAMP_T{T}_lam{lam}_{untiedf}_{coordf}_{exp_id}"
                        .format (T=config.T, lam=config.lam, untiedf=untiedf,
                                 coordf=coordf, exp_id=config.exp_id))
        from models.LAMP import LAMP
        model = LAMP (kwargs ['A'], T=config.T, lam=config.lam,
                      untied=config.untied, coord=config.coord,
                      scope=config.scope)

    if config.net == 'LIHT' :
        """LIHT"""
        from models.LIHT import LIHT
        model = LIHT (p, T=config.T, lam=config.lam, y_=p.y_ , x0_=None ,
                      untied=config.untied , cord=config.coord)

    if config.net == 'LISTA_cp' :
        """LISTA-CP"""
        config.model = ("LISTA_cp_T{T}_lam{lam}_{untiedf}_{coordf}_{exp_id}"
                        .format (T=config.T, lam=config.lam, untiedf=untiedf,
                                 coordf=coordf, exp_id=config.exp_id))
        from models.LISTA_cp import LISTA_cp
        model = LISTA_cp (kwargs ['A'], T=config.T, lam=config.lam,
                          untied=config.untied, coord=config.coord,
                          scope=config.scope)

    if config.net == 'LISTA_ss' :
        """LISTA-SS"""
        config.model = ("LISTA_ss_T{T}_lam{lam}_p{p}_mp{mp}_"
                        "{untiedf}_{coordf}_{exp_id}"
                        .format (T=config.T, lam=config.lam, p=config.percent,
                                 mp=config.max_percent, untiedf=untiedf,
                                 coordf=coordf, exp_id=config.exp_id))
        from models.LISTA_ss import LISTA_ss
        model = LISTA_ss (kwargs ['A'], T=config.T, lam=config.lam,
                          percent=config.percent, max_percent=config.max_percent,
                          untied=config.untied , coord=config.coord,
                          scope=config.scope)

    if config.net == 'LISTA_cpss' :
        """LISTA-CPSS"""
        config.model = ("LISTA_cpss_T{T}_lam{lam}_p{p}_mp{mp}_"
                        "{untiedf}_{coordf}_{exp_id}"
                        .format (T=config.T, lam=config.lam, p=config.percent,
                                 mp=config.max_percent, untiedf=untiedf,
                                 coordf=coordf, exp_id=config.exp_id))
        from models.LISTA_cpss import LISTA_cpss
        model = LISTA_cpss (kwargs ['A'], T=config.T, lam=config.lam,
                            percent=config.percent, max_percent=config.max_percent,
                            untied=config.untied , coord=config.coord,
                            scope=config.scope)

    if config.net == 'TiLISTA':
        """TiLISTA"""
        config.model = ("TiLISTA_T{T}_lam{lam}_p{p}_mp{mp}_"
                        "{coordf}_{exp_id}"
                        .format (T=config.T, lam=config.lam,
                                 p=config.percent, mp=config.max_percent,
                                 coordf=coordf, exp_id=config.exp_id))
        from models.TiLISTA import TiLISTA
        # Note that TiLISTA is just LISTA-CPSS with tied weight in all layers.
        model = TiLISTA(kwargs['A'], T=config.T, lam=config.lam,
                        percent=config.percent, max_percent=config.max_percent,
                        coord=config.coord, scope=config.scope)

    if config.net == "ALISTA":
        """ALISTA"""
        config.model = ("ALISTA_T{T}_lam{lam}_p{p}_mp{mp}_"
                        "{W}_{coordf}_{exp_id}"
                        .format(T=config.T, lam=config.lam,
                                p=config.percent, mp=config.max_percent,
                                W=os.path.basename(config.W),
                                coordf=coordf, exp_id=config.exp_id))
        W = np.load(config.W)
        print("Pre-calculated weight W loaded from {}".format(config.W))
        from models.ALISTA import ALISTA
        model = ALISTA(kwargs['A'], T=config.T, lam=config.lam, W=W,
                       percent=config.percent, max_percent=config.max_percent,
                       coord=config.coord, scope=config.scope)

    if config.net == 'LISTA_cs':
        """LISTA-CS"""
        config.model = ("LISTA_cs_T{T}_lam{lam}_llam{llam}_"
                        "{untiedf}_{coordf}_{exp_id}"
                        .format (T=config.T, lam=config.lam, llam=config.lasso_lam,
                                 untiedf=untiedf, coordf=coordf,
                                 exp_id=config.exp_id))
        from models.LISTA_cs import LISTA_cs
        model = LISTA_cs (kwargs ['Phi'], kwargs ['D'], T=config.T,
                          lam=config.lam, untied=config.untied,
                          coord=config.coord, scope=config.scope)

    if config.net == 'LISTA_ss_cs' :
        """LISTA-SS-CS"""
        config.model = ("LISTA_ss_cs_T{T}_lam{lam}_p{p}_mp{mp}_llam{llam}_"
                        "{untiedf}_{coordf}_{exp_id}"
                        .format (T=config.T, lam=config.lam, p=config.percent,
                                 mp=config.max_percent, llam=config.lasso_lam,
                                 untiedf=untiedf, coordf=coordf,
                                 exp_id=config.exp_id))
        from models.LISTA_ss_cs import LISTA_ss_cs
        model = LISTA_ss_cs (kwargs ['Phi'], kwargs ['D'], T=config.T,
                             lam=config.lam, percent=config.percent,
                             max_percent=config.max_percent,
                             untied=config.untied, coord=config.coord,
                             scope=config.scope)

    if config.net == 'LISTA_cpss_cs' :
        """LISTA-CPSS-CS"""
        config.model = ("LISTA_cpss_cs_T{T}_lam{lam}_p{p}_mp{mp}_llam{llam}_"
                        "{untiedf}_{coordf}_{exp_id}"
                        .format (T=config.T, lam=config.lam, p=config.percent,
                                 mp=config.max_percent, llam=config.lasso_lam,
                                 untiedf=untiedf, coordf=coordf,
                                 exp_id=config.exp_id))
        from models.LISTA_cpss_cs import LISTA_cpss_cs
        model = LISTA_cpss_cs (kwargs ['Phi'], kwargs ['D'], T=config.T,
                               lam=config.lam, percent=config.percent,
                               max_percent=config.max_percent,
                               untied=config.untied, coord=config.coord,
                               scope=config.scope)


    config.modelfn = os.path.join (config.expbase, config.model)
    config.resfn   = os.path.join (config.resbase, config.model)
    print ("model disc:", config.model)

    return model


############################################################
######################   Training    #######################
############################################################

def run_train (config) :
    if config.task_type == 'sc':
        run_sc_train (config)
    elif config.task_type == 'cs':
        run_cs_train (config)


def run_sc_train (config) :
    """Load problem."""
    if not os.path.exists (config.probfn):
        raise ValueError ("Problem file not found.")
    else:
        p = problem.load_problem (config.probfn)

    """Set up model."""
    model = setup_model (config, A=p.A)

    """Set up input."""
    config.SNR = np.inf if config.SNR == 'inf' else float (config.SNR)
    y_, x_, y_val_, x_val_ = (
        train.setup_input_sc (
            config.test, p, config.tbs, config.vbs, config.fixval,
            config.supp_prob, config.SNR, config.magdist, **config.distargs))

    """Set up training."""
    stages = train.setup_sc_training (
            model, y_, x_, y_val_, x_val_, None,
            config.init_lr, config.decay_rate, config.lr_decay)


    tfconfig = tf.ConfigProto (allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session (config=tfconfig) as sess:
        # graph initialization
        sess.run (tf.global_variables_initializer ())

        # start timer
        start = time.time ()

        # train model
        model.do_training (sess, stages, config.modelfn, config.scope,
                           config.val_step, config.maxit, config.better_wait)

        # end timer
        end = time.time ()
        elapsed = end - start
        print ("elapsed time of training = " + str (timedelta (seconds=elapsed)))


def run_cs_train (config) :
    """Load dictionary and sensing matrix."""
    Phi = np.load (config.sensing) ['A']
    D   = np.load (config.dict)

    """Set up model."""
    model = setup_model (config, Phi=Phi, D=D)

    """Set up inputs."""
    y_, f_, y_val_, f_val_ = train.setup_input_cs(config.train_file,
                                                  config.val_file,
                                                  config.tbs, config.vbs)

    """Set up training."""
    stages = train.setup_cs_training (
        model, y_, f_, y_val_, f_val_, None, config.init_lr, config.decay_rate,
        config.lr_decay, config.lasso_lam)


    """Start training."""
    tfconfig = tf.ConfigProto (allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session (config=tfconfig) as sess:
        # graph initialization
        sess.run (tf.global_variables_initializer ())

        # start timer
        start = time.time ()

        # train model
        model.do_training (sess, stages, config.modelfn, config.scope,
                           config.val_step, config.maxit, config.better_wait)

        # end timer
        end = time.time ()
        elapsed = end - start
        print ("elapsed time of training = " + str (timedelta (seconds=elapsed)))


############################################################
######################   Testing    ########################
############################################################

def run_test (config):
    if config.task_type == 'sc':
        run_sc_test (config)
    elif config.task_type == 'cs':
        run_cs_test (config)

def run_sc_test (config) :
    """
    Test model.
    """

    """Load problem."""
    if not os.path.exists (config.probfn):
        raise ValueError ("Problem file not found.")
    else:
        p = problem.load_problem (config.probfn)

    """Load testing data."""
    xt = np.load (config.xtest)
    """Set up input for testing."""
    config.SNR = np.inf if config.SNR == 'inf' else float (config.SNR)
    input_, label_ = (
        train.setup_input_sc (config.test, p, xt.shape [1], None, False,
                              config.supp_prob, config.SNR,
                              config.magdist, **config.distargs))

    """Set up model."""
    model = setup_model (config , A=p.A)
    xhs_ = model.inference (input_, None, False)

    """Create session and initialize the graph."""
    tfconfig = tf.ConfigProto (allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session (config=tfconfig) as sess:
        # graph initialization
        sess.run (tf.global_variables_initializer ())
        # load model
        model.load_trainable_variables (sess , config.modelfn)

        nmse_denom = np.sum (np.square (xt))
        supp_gt = xt != 0

        lnmse  = []
        lspar  = []
        lsperr = []
        lflspo = []
        lflsne = []

        # test model
        for xh_ in xhs_ :
            xh = sess.run (xh_ , feed_dict={label_:xt})

            # nmse:
            loss = np.sum (np.square (xh - xt))
            nmse_dB = 10.0 * np.log10 (loss / nmse_denom)
            print (nmse_dB)
            lnmse.append (nmse_dB)

            supp = xh != 0.0
            # intermediate sparsity
            spar = np.sum (supp , axis=0)
            lspar.append (spar)

            # support error
            sperr = np.logical_xor(supp, supp_gt)
            lsperr.append (np.sum (sperr , axis=0))

            # false positive
            flspo = np.logical_and (supp , np.logical_not (supp_gt))
            lflspo.append (np.sum (flspo , axis=0))

            # false negative
            flsne = np.logical_and (supp_gt , np.logical_not (supp))
            lflsne.append (np.sum (flsne , axis=0))

    res = dict (nmse=np.asarray  (lnmse),
                spar=np.asarray  (lspar),
                sperr=np.asarray (lsperr),
                flspo=np.asarray (lflspo),
                flsne=np.asarray (lflsne))

    np.savez (config.resfn , **res)
    # end of test


def run_cs_test (config) :
    """Load dictionary and sensing matrix."""
    Phi = np.load (config.sensing) ['A']
    D   = np.load (config.dict)

    # loading compressive sensing settings
    M = Phi.shape [0]
    F = Phi.shape [1]
    N = D.shape [1]
    assert M == config.M and F == config.F and N == config.N
    patch_size = int (np.sqrt (F))
    assert patch_size ** 2 == F


    """Set up model."""
    model = setup_model (config, Phi=Phi, D=D)

    """Inference."""
    y_ = tf.placeholder (shape=(M, None), dtype=tf.float32)
    _, fhs_ = model.inference (y_, None)


    """Start testing."""
    tfconfig = tf.ConfigProto (allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session (config=tfconfig) as sess:

        # graph initialization
        sess.run (tf.global_variables_initializer ())
        # load model
        model.load_trainable_variables (sess , config.modelfn)

        # calculate average NMSE and PSRN on test images
        test_dir = './data/test_images/'
        test_files = os.listdir (test_dir)
        avg_nmse = 0.0
        avg_psnr = 0.0
        overlap = 0
        stride = patch_size - overlap
        if 'joint' in config.net :
            D = sess.run (model.D_)
        for test_fn in test_files :
            # read in image
            test_fn = os.path.join (test_dir, test_fn)
            test_im, H, W, test_im_pad, H_pad, W_pad = \
                    imread_CS_py (test_fn, patch_size, stride)
            test_fs = img2col_py (test_im_pad, patch_size, stride)

            # remove dc from features
            test_dc = np.mean (test_fs, axis=0, keepdims=True)
            test_cfs = test_fs - test_dc
            test_cfs = np.asarray (test_cfs) / 255.0

            # sensing signals
            test_ys = np.matmul (Phi, test_cfs)
            num_patch = test_ys.shape [1]

            rec_cfs = sess.run (fhs_ [-1], feed_dict={y_: test_ys})
            print (rec_cfs.shape)
            rec_fs  = rec_cfs * 255.0 + test_dc

            # patch-level NMSE
            patch_err = np.sum (np.square (rec_fs - test_fs))
            patch_denom = np.sum (np.square (test_fs))
            avg_nmse += 10.0 * np.log10 (patch_err / patch_denom)

            rec_im = col2im_CS_py (rec_fs, patch_size, stride,
                                   H, W, H_pad, W_pad)

            # image-level PSNR
            image_mse = np.mean (np.square (rec_im - test_im))
            avg_psnr += 10.0 * np.log10 (255.**2 / image_mse)

    num_test_ims = len (test_files)
    print ('Average Patch-level NMSE is {}'.format (avg_nmse / num_test_ims))
    print ('Average Image-level PSNR is {}'.format (avg_psnr / num_test_ims))

    # end of cs_testing

############################################################
#######################    Main    #########################
############################################################

def main ():
    # parse configuration
    config, _ = get_config()
    # set visible GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

    if config.test:
        run_test (config)
    else:
        run_train (config)
    # end of main

if __name__ == "__main__":
    main ()

