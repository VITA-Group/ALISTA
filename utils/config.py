#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
config.py
author: xhchrn
        chernxh@tamu.edu

Set up experiment configuration using argparse library.
"""

import os
import sys
import datetime
import argparse
from math import ceil

def str2bool(v):
    return v.lower() in ('true', '1')

parser = argparse.ArgumentParser()

# Network arguments
net_arg = parser.add_argument_group('net')
net_arg.add_argument(
    '-n', '--net', type=str,
    help='Network name.')
net_arg.add_argument(
    '-T', '--T', type=int, default=16,
    help="Number of layers of LISTA.")
net_arg.add_argument(
    '-eT', '--eT', type=int, default=4,
    help="Number of layers of encoders.")
net_arg.add_argument(
    '-dT', '--dT', type=int, default=16,
    help="Number of layers of decoders.")
net_arg.add_argument(
    '-p', '--percent', type=float, default=0.8,
    help="Percent of entries to be selected as support in each layer.")
net_arg.add_argument(
    '-maxp', '--max_percent', type=float, default=0.0,
    help="Maximum percentage of entries to be selectedas support in each layer.")
net_arg.add_argument(
    '-l', '--lam', type=float, default=0.4,
    help="Initial lambda in LISTA solvers.")
net_arg.add_argument(
    '-u', '--untied', action='store_true',
    help="Whether weights are untied between layers.")
net_arg.add_argument(
    '-c', '--coord', action='store_true',
    help="Whether use independent vector thresholds.")
net_arg.add_argument(
    '-W', '--W', type=str, default="",
    help="Pretrained weights for fbss models.")
net_arg.add_argument(
    "-eta", "--eta", type=float, default=1e-3,
    help="The initial step size in the projected gradient descent for encoding weight matrices.")
net_arg.add_argument(
    "-Q", "--Q", type=str, default=None,
    help="The matrix that is used to reweight the loss function in encoder training.")
net_arg.add_argument(
    '-sc', '--scope', type=str, default="",
    help="Scope name of the model.")

# Problem arguments
prob_arg = parser.add_argument_group('prob')
prob_arg.add_argument(
    '-M', '--M', type=int, default=250,
    help="Dimension of measurements.")
prob_arg.add_argument(
    '-N', '--N', type=int, default=500,
    help="Dimension of sparse codes.")
prob_arg.add_argument(
    '-F', '--F', type=int, default=256,
    help='Number of features of extracted patches.')
prob_arg.add_argument(
    '-sr', '--sample_rate', type=int, default=None,
    help="Sampling rate in compressive sensing experiments.")
prob_arg.add_argument(
    '-P', '--pnz', type=float, default=0.1,
    help="Percent of nonzero entries in sparse codes.")
prob_arg.add_argument(
    '-S', '--SNR', type=str, default='inf',
    help="Strength of noises in measurements.")
prob_arg.add_argument(
    '-C', '--con_num', type=float, default=0.0,
    help="Condition number of measurement matrix. 0 for no modification on condition number.")
prob_arg.add_argument(
    '-CN', '--col_normalized', type=str2bool, default=True,
    help="Flag of whether normalize the columns of the dictionary or sensing matrix.")
prob_arg.add_argument(
    '-task', '--task_type', type=str, default='sc',
    help='Task type, in [`sc`, `cs`, `denoise`].')
prob_arg.add_argument(
    '-llam', '--lasso_lam', type=float, default=0.2,
    help='The weight of l1 norm term `labmda` in LASSO.')
# Conv problem arguments
prob_arg.add_argument(
    '-cd', '--conv_d', type=int, default=3,
    help="The size of kernels in a convolutional dictionary.")
prob_arg.add_argument(
    '-cm', '--conv_m', type=int, default=100,
    help="The number of kernels in a convolutional dictionary.")
prob_arg.add_argument(
    '-clam', '--conv_lam', type=float, default=0.05,
    help="The weight in the objective function used to learn the convolutional dictioanry.")
prob_arg.add_argument(
    '-ca', '--conv_alpha', type=float, default=0.1,
    help="The initial step size for convolutional ISTA.")

"""Training arguments."""
train_arg = parser.add_argument_group('train')
train_arg.add_argument(
    '-lr', '--init_lr', type=float, default=5e-4,
    help="Initial learning rate.")
train_arg.add_argument(
    '-tbs', '--tbs', type=int, default=64,
    help="Training batch size.")
train_arg.add_argument(
    '-vbs', '--vbs', type=int, default=1000,
    help="Validation batch size.")
train_arg.add_argument(
    '-fixval', '--fixval', type=str2bool, default=True,
    help="Flag of whether we fix a validation set.")
train_arg.add_argument(
    '-supp_prob', '--supp_prob', type=str, default=None,
    help="The probability distribution of support we use in trianing.")
train_arg.add_argument(
    '-magdist', '--magdist', type=str, default='normal',
    help="Type of the magnitude distribution.")
train_arg.add_argument(
    '-nmean', '--magnmean', type=float, default=0.0,
    help="The expectation of Gaussian that we use to sample magnitudes.")
train_arg.add_argument(
    '-nstd', '--magnstd', type=float, default=1.0,
    help="The standard deviation of Gaussain we use to sample magnitudes.")
train_arg.add_argument(
    '-bp', '--magbp', type=float, default=0.5,
    help="The probability that the magnitudes take value `magbv0` when they are"
         "sampled from Bernoulli.")
train_arg.add_argument(
    '-bv0', '--magbv0', type=float, default=1.0,
    help="The value that the magnitudes take with probability `magbp` when they"
         "are sampled from Bernoulli.")
train_arg.add_argument(
    '-bv1', '--magbv1', type=float, default=1.0,
    help="The value that the magnitudes take with probability `1-magbp` when"
         "they are sampled from Bernoulli.")
train_arg.add_argument(
    '-sigma', '--sigma', type=float, default=20.0,
    help="The standard deviation of image noises for denoise task.")
train_arg.add_argument(
    "-hc", "--height_crop", type=int, default=321,
    help="The height after cropping for BSD500 denoising experiments.")
train_arg.add_argument(
    "-wc", "--width_crop", type=int, default=321,
    help="The width after cropping for BSD500 denoising experiments.")
train_arg.add_argument(
    "-epoch", "--num_epochs", type=int, default=20,
    help="The number of training epochs for denoise experiments.\n"
         "`-1` means infinite number of epochs.")
train_arg.add_argument(
    '-dr', '--decay_rate', type=float, default=0.3,
    help="Learning rate decaying rate after training each layer.")
train_arg.add_argument(
    '-ld', '--lr_decay', type=str, default='0.2,0.02',
    help="Learning rate decaying rate after training each layer.")
train_arg.add_argument(
    '-vs', '--val_step', type=int, default=10,
    help="Interval of validation in training.")
train_arg.add_argument(
    '-mi', '--maxit', type=int, default=200000,
    help="Max number iteration of each stage.")
train_arg.add_argument(
    '-bw', '--better_wait', type=int, default=4000,
    help="Waiting time before jumping to next stage.")


# Experiments arguments
exp_arg = parser.add_argument_group('exp')
exp_arg.add_argument(
    '-id', '--exp_id', type=int, default=0,
    help="ID of the experiment/model.")
exp_arg.add_argument(
    '-ef', '--exp_folder', type=str, default='./experiments',
    help="Root folder for problems and momdels.")
exp_arg.add_argument(
    '-rf', '--res_folder', type=str, default='./results',
    help="Root folder where test results are saved.")
exp_arg.add_argument(
    '-pf', '--prob_folder', type=str, default='',
    help="Subfolder in exp_folder for a specific setting of problem.")
exp_arg.add_argument(
    '--prob', type=str, default='prob.npz',
    help="Problem file name in prob_folder.")
exp_arg.add_argument(
    '-se', '--sensing', type=str, default=None,
    help="Sensing matrix file. Instance of Problem class.")
exp_arg.add_argument(
    '-dc', '--dict', type=str, default=None,
    help="Dictionary file. Numpy array instance stored as npy file.")
exp_arg.add_argument(
    '-df', '--data_folder', type=str, default=None,
    help="Folder where the tfrecords datasets are stored.")
exp_arg.add_argument(
    '-tf', '--train_file', type=str, default='train.tfrecords',
    help="File name of tfrecords file of training data for cs exps.")
exp_arg.add_argument(
    '-vf', '--val_file', type=str, default='val.tfrecords',
    help="File name of tfrecords file of validation data for cs exps.")
exp_arg.add_argument(
    '-col', '--column', type=str2bool, default=False,
    help="Flag of whether column-based model is used.")
exp_arg.add_argument(
    '-t', '--test', action='store_true',
    help="Flag of training or testing models.")
exp_arg.add_argument(
    '-np', '--norm_patch', type=str2bool, default=False,
    help="Flag of normalizing patches in training and testing.")
exp_arg.add_argument(
    '-xt', '--xtest', type=str, default='./data/xtest_n500_p10.npy',
    help='Default test x input for simulation experiments.')
exp_arg.add_argument(
    "-td", "--test_dir", type=str, default="./data/test_images",
    help="Directory that holds testing images for compreessive sensing and "
         "denoising experiments.")
exp_arg.add_argument(
    '-g', '--gpu', type=str, default='0',
    help="ID's of allocated GPUs.")
################################################
####### Arguments for robust experiments #######
################################################
parser.add_argument(
    "-ps_max", "--psigma_max", type=float, default=2e-2,
    help="Maximum standard deviation for perturbations in dictionaries.")
parser.add_argument(
    "-psteps", "--psteps", type=int, default=5,
    help="The number of steps for curriculum learning that gradually increases"
         "the level of perturbations.")
parser.add_argument(
    "-ms", "--msigma", type=float, default=0.0,
    help="Standard deviation for measurement noisy in robustness experiments.")
parser.add_argument(
    "-eps", "--encoder_psigma", type=float, default=1e-2,
    help="The standard deviation for perturbations in dictionaries for encoder pre-training.")
parser.add_argument(
    "-elr", "--encoder_lr", type=float, default=1e-6,
    help="The initial learning rate for encoders.")
parser.add_argument(
    "-eplr", "--encoder_pre_lr", type=float, default=1e-4,
    help="The initial learning rate for pre-training encoders.")
parser.add_argument(
    "-dlr", "--decoder_lr", type=float, default=1e-4,
    help="The initial learning rate for decoders.")
# parser.add_argument(
#     "-dlr", "--decoder_pre_lr", type=float, default=1e-4,
#     help="The initial learning rate for pre-training decoders.")
parser.add_argument(
    "-eb", "--encoder_Binit", type=str, default="default",
    help="The initial weight matrices in the encoders.")
parser.add_argument(
    "-el", "--encoder_loss", type=str, default="rel2",
    help="The loss type of encoder pretraining.")
parser.add_argument(
    "-escope", "--encoder_scope", type=str, default="",
    help="The scope name for encoders.")
parser.add_argument(
    "-dscope", "--decoder_scope", type=str, default="",
    help="The scope name for decoders.")
parser.add_argument(
    "-eid", "--encoder_id", type=int, default=0,
    help="The model id for encoders.")
parser.add_argument(
    "-did", "--decoder_id", type=int, default=0,
    help="The model id for decoders.")
parser.add_argument(
    "-Abs", "--Abs", type=int, default=4,
    help="The number of perturbed dictionaries sampled in each robustness training step.")
parser.add_argument(
    "-xbs", "--xbs", type=int, default=16,
    help="The number of sparse vectors sampled in each robustness training step.")
# parser.add_argument(
#     "-maxit", "--maxit", type=int, default=50000,
#     help="Max iterations in each stage in robustness experiments.")


def get_config():
    config, unparsed = parser.parse_known_args()

    """
    Check validity of arguments.
    """
    # check if a network model is specified
    if config.net == None:
        raise ValueError('no model specified')

    # set experiment path and folder
    if not os.path.exists(config.exp_folder):
        os.mkdir(config.exp_folder)


    """Experiments and results base folder."""
    if config.prob_folder == "":
        if config.task_type in ['sc', "encoder", 'robust']:
            config.prob_folder = ('m{}_n{}_k{}_p{}_s{}'.format(
                config.M, config.N, config.con_num, config.pnz, config.SNR))
        elif config.task_type == 'cs':
            # check problem folder: dictionary and sensing matrix
            config.prob_folder = ('cs_bsd_d{}-{}'.format(config.F, config.N))
        elif config.task_type == "denoise":
            config.prob_folder = ("denoise_bsd500_d{d}_m{m}_lam{lam}".format(
                d=config.conv_d, m=config.conv_m, lam=config.conv_lam))

    # make experiment base path and results base path
    setattr(config, 'expbase', os.path.join(config.exp_folder,
                                            config.prob_folder))
    setattr(config, 'resbase', os.path.join(config.res_folder,
                                            config.prob_folder))
    if not os.path.exists(config.expbase):
        os.mkdir(config.expbase)
    if not os.path.exists(config.resbase):
        os.mkdir(config.resbase)

    if config.task_type == 'cs':
        if not config.sample_rate is None:
            config.M = ceil(config.F * config.sample_rate / 100.0)
        else:
            config.sample_rate = ceil(config.M / config.F * 100.0)
        config.expbase = os.path.join(config.expbase, "r%d"%config.sample_rate)
        config.resbase = os.path.join(config.resbase, "r%d"%config.sample_rate)

    """
    Problem file for sparse coding task.
    Data folder, dictionary and sensing file for compressive sensing task.
    """
    if config.task_type == 'sc':
        setattr(config, 'probfn' , os.path.join(config.expbase, config.prob))
    # Data folder, dictionary and sensing matrix location for cs experiments.
    elif config.task_type == 'cs' and not config.test:
        # check data files, dictionary and sensing matrix
        if config.train_file is None:
            raise ValueError("Please provide a training tfrecords file for CS exp!")
        if config.val_file is None:
            raise ValueError("Please provide a validation tfrecords file for CS exp!")
        if config.dict is None:
            raise ValueError("Please provide a dictionary for CS exp!")
        if config.sensing is None:
            raise ValueError("Please provide a sensing matrix for CS exp!")

        if not os.path.exists(config.train_file) :
            raise ValueError('No training data tfrecords file found.')
        if not os.path.exists(config.val_file) :
            raise ValueError('No validation data tfrecords file found')
        if not os.path.exists(config.sensing) :
            raise ValueError('No sensing matrix file found')
        if not os.path.exists(config.dict) :
            raise ValueError('No dictionary matrix file found')
    # Data folder for denoise experiments.
    elif config.task_type == 'denoise':
        setattr(config, 'probfn' , os.path.join(config.expbase, config.prob))
        if config.data_folder is None:
            raise ValueError("Please provide a dataset directory tfrecords file for denoise exp!")
        if config.train_file is None:
            raise ValueError("Please provide a training tfrecords file for denoise exp!")
        if config.val_file is None:
            raise ValueError("Please provide a validation tfrecords file for denoise exp!")
        if not os.path.exists(os.path.join(config.data_folder, config.train_file)) :
            raise ValueError('No training data tfrecords file found.')
        if not os.path.exists(os.path.join(config.data_folder, config.val_file)) :
            raise ValueError('No validation data tfrecords file found')
    elif config.task_type == 'robust':
        setattr(config, 'probfn' , os.path.join(config.expbase, config.prob))
    elif config.task_type == 'encoder':
        setattr(config, 'probfn' , os.path.join(config.expbase, config.prob))


    # lr_decay
    config.lr_decay = tuple([float(decay) for decay in config.lr_decay.split(',')])

    """Noise standard deviation for denoise experiments:
    sigma -> standard deviation."""
    if config.task_type == 'denoise':
        config.denoise_std = config.sigma / 255.0

    """Support and magnitudes distribution settings for sparse coding task."""
    if config.task_type == 'sc':
        # supp_prob
        if not config.supp_prob is None:
            try:
                config.supp_prob = float(config.supp_prob)
            except ValueError:
                import numpy as np
                config.supp_prob = np.load(config.supp_prob)

        """Magnitudes distribution of sparse codes."""
        if config.magdist == 'normal':
            config.distargs = dict(mean=config.magnmean,
                                   std=config.magnstd)
        elif config.magdist == 'bernoulli':
            config.distargs = dict(p=config.magbp,
                                   v0=config.magbv0,
                                   v1=config.magbv1)

    return config, unparsed

