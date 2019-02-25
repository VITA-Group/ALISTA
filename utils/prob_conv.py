#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
file  : prob_conv.py
author: Xiaohan Chen
email : chernxh@tamu.edu
data  : 2019-02-23
"""

import os
import argparse
import numpy as np
import numpy.linalg as la
from scipy.io import savemat, loadmat
from scipy.signal import convolve2d

def str2bool(v):
    return v.lower() in ('true', '1')

class ProblemConv(object):

    """
    Define the model of convolutional sparse coding.

    In every problem, we define:
        :fs   : Numpy array of size (fh, fw, fn). Convolutional dictionary.
        :fh   : Integer. Height of filters.
        :fw   : Integer. Width of filters.
        :fn   : Integer. Number of filters.
        :pnz  : Float. Percentage of entries in feature maps that are not zeros.
        :SNR  : Float. Noise level in measurements.
    """

    def __init__(self):
        pass

    def build_prob(self, fs , pnz=0.1, SNR=40.0):
        self._fs   = fs
        self._ft   = np.rot90 (self._fs, axes=(0,1), k=2)
        self._fh   = self._fs.shape [0]
        self._fw   = self._fs.shape [1]
        self._fn   = self._fs.shape [2]
        self._pnz  = pnz
        self._SNR  = SNR

        # TODO: how to calculate noise level? SNR -> noise variace?
        self._noise_var = 1e-50
        # self.noise_var = pnz * self.N / self.M * np.power (10.0 , -SNR/10.0 )

        # self.yval, self.xval = self.gen_samples( self.L )

    def measure (self, features, noise_var=None):
        """
        Convolve feature map using self._fs.

        :features : Numpy array of size (bs, h + fh - 1, w + fw - 1, fn)
        :noise_var: Float. The variance of the noise.

        Only consider noiseless setting now.
        """
        if noise_var == None:
            noise_var = self._noise_var

        bs, fmh, fmw, fmn = features.shape
        assert fmn == self._fn  # the number of feature maps should be same as
                                # the number of filters
        ih = fmh - self._fh + 1 # image height
        iw = fmw - self._fw + 1 # image width

        # do convolution
        convs = []
        for feature in features:
            conv = np.zeros (shape=(ih, iw), dtype=np.float32)
            for i in range (self._fn):
                # NOTE: Here we use the transpose of the filters to convolve
                # because numpy will first transpose the filters before
                # convolution.
                conv += convolve2d (feature  [:,:,i],
                                    self._ft [:,:,i],
                                    mode='valid')
            convs.append (conv)
        convs = np.asarray (convs, dtype=np.float32)

        # add noises
        noise = np.random.normal (size=convs.shape, scale=np.sqrt(noise_var)).\
                    astype (np.float32)

        return convs + noise


    def gen_samples(self, bs, ih, iw, pnz=None, SNR=None, probability=None):
        """TODO: Docstring for gen_samples.
        Generate samples (y, x) in current problem setting.

        :bs: Integer. Batch size, the number of images to be generated.
        :ih: Integer. Height of the generated images.
        :iw: Integer. Width of the generated images.
        :pnz: Float. Percentage in decimal of the sparsity of feature maps.
        :SNR: Float. Signal-to-Noise ratio in this measurement.
        :probability: Float or numpy.ndarray. Probability map of the probability
                      that entries in the feature maps are none-zero.
        :returns: TODO

        """
        if pnz is None:
            pnz = self._pnz

        if SNR is None:
            noise_var = self._noise_var
        # TODO: exetnd to noisy case
        # else:
        #     noise_var = (self.pnz * self.N / self.M *
        #                  np.power (10.0 , -SNR/10.0 ))
        noise_var = np.max ([noise_var , 1e-50])

        fmh, fmw, fmn = ih + self._fh - 1, iw + self._fw - 1, self._fn

        if probability is None:
            probability = pnz
        else:
            assert probability.shape == (fmh, fmw, fmn)
            assert np.abs (np.sum (probability) - fmh * fmw * fmn * pnz) < 1

        bernoulli = np.random.uniform (size=(bs, fmh, fmw, fmn)) <= probability
        bernoulli = bernoulli.astype (np.float32)
        features  = (bernoulli * (np.random.normal (size=(bs, fmh, fmw, fmn))
                            .astype (np.float32)))

        images = self.measure (features , noise_var)
        return images, features

    def save(self, path, ftype='npz'):
        """
        Save current problem settings to npz file or mat file.
        """
        D = dict(fs=self._fs,
                 ft=self._ft,
                 fh=self._fh,
                 fw=self._fw,
                 fn=self._fn,
                 pnz=self._pnz,
                 SNR=self._SNR,
                 noise_var=self._noise_var)
        if path[-4:] != '.' + ftype:
            path = path + '.' + ftype

        if ftype == 'npz':
            np.savez( path, **D )
        elif ftype == 'mat':
            savemat( path, D, oned_as='column' )
        else:
            raise ValueError ('invalid file type {}'.format (ftype))


    def read (self, fname):
        """
        Read saved problem from a npz/mat file.
        """
        if not os.path.exists( fname ):
            raise ValueError('saved problem file {} not found'.format( fname ))
        if fname[-4:] == '.npz':
            # read from saved npz file
            D = np.load (fname)
        elif fname[-4:] == '.mat':
            # read from saved mat file
            D = loadmat (fname)
        else:
            raise ValueError('invalid file type; npz or mat file required')

        if not 'fs' in D.keys():
            raise ValueError('invalid input file; filters fs missing')

        for k, v in D.items():
            setattr (self, "_"+k, v)

        print( "problem {} successfully loaded".format( fname ) )


def random_fs (shape):
    """
    Randomly sample filters from i.i.d. Gaussian and then normalize each filter.

    :shape: Tuple of integers. Assume it has the form of (fh, fw, fn).
    :returns:
        fs: numpy.ndarray of shape (fh, fw, fn).

    """
    if len (shape) != 3:
        raise ValueError ("The shape of filters should be of the form of"
                          "(height, width, channels).")

    fs = np.random.normal (size=shape).astype(np.float32)

    # normalization
    norms = np.sqrt (np.sum (np.square (fs), axis=(0,1)))
    return fs / norms


def setup_problem(fs, pnz, SNR):
    # create and build problem for conv sparse coding
    prob = ProblemConv ()
    prob.build_prob (fs, pnz, SNR)

    return prob


def load_problem (fname):
    prob = ProblemConv()
    prob.read (fname)
    return prob

parser = argparse.ArgumentParser()
parser.add_argument(
    '-cd', '--conv_d', type=int, default=3,
    help="The size of kernels in a convolutional dictionary.")
parser.add_argument(
    '-cm', '--conv_m', type=int, default=100,
    help="The number of kernels in a convolutional dictionary.")
parser.add_argument(
    '-clam', '--conv_lam', type=float, default=0.05,
    help="The weight in the objective function used to learn the convolutional dictioanry.")
parser.add_argument(
    '-L', '--L', type=int, default=0,
    help="Number of samples for validation (deprecated. please use default).")
parser.add_argument(
    '-P', '--pnz', type=float, default=0.001,
    help="Percent of nonzero entries in sparse codes.")
parser.add_argument(
    '-S', '--SNR', type=str, default='inf',
    help="Strength of noises in measurements.")
parser.add_argument(
    '-C', '--con_num', type=float, default=0.0,
    help="Condition number of measurement matrix. 0 for no modification on condition number.")
parser.add_argument(
    '-CN', '--col_normalized', type=str2bool, default=True,
    help="Flag of whether normalize the columns of the dictionary or sensing matrix.")
parser.add_argument(
    "-ld", "--load_dict", type=str, default=None,
    help="Path to the convolutional dictionary to be loaded.")
parser.add_argument(
    '-ef', '--exp_folder', type=str, default='./experiments',
    help="Root folder for problems and momdels.")
parser.add_argument(
    "-pfn", "--prob_file", type=str, default="prob.npz",
    help="The (base) file name of problem file.")

if __name__ == "__main__":
    config, unparsed = parser.parse_known_args()
    if not config.load_dict is None:
        if config.load_dict.endswith(".npy"):
            D = np.load(config.load_dict)
            print("matrix loaded from {}. will be used to generate the problem"
                  .format(config.load_dict))
        elif config.load_dict.endswith(".mat"):
            import scipy.io as sio
            D = sio.loadmat(config.load_dict)['D']
            # NOTE: here we rotate the filters in `D` with 180 degrees because
            #       the conv2d in TensorFlow is actually the correlation
            #       operation in the convolutional dictionary learning
            #       terminology.
            D = np.rot90(D, axes=(0,1), k=2)
        else:
            raise ValueError("invalid file {}".format(config.load_dict))
        config.conv_d, config.conv_m = D.shape[0], D.shape[2]
    else:
        shape = (config.conv_d, config.conv_d, config.conv_m)
        config.conv_lam = 0.0
        D = random_fs(shape)
    prob_desc = ('denoise_d{}_m{}_lam{}'.format(config.conv_d, config.conv_m,
                                                config.conv_lam))
    prob_folder = os.path.join(config.exp_folder, prob_desc)
    if not os.path.exists(prob_folder):
        os.makedirs(prob_folder)
    out_file = os.path.join(config.exp_folder, prob_desc, config.prob_file)
    if os.path.exists(out_file):
        raise ValueError("specified problem file {} already exists".format(out_file))
    if config.SNR == "inf":
        SNR = np.inf
    else:
        try:
            SNR = float(config.SNR)
        except Exception as e:
            raise ValueError("invalid SNR. use 'inf' or a float number.")
    p = setup_problem(D, config.pnz, SNR)
    p.save(out_file, ftype="npz")
    print("problem saved to {}".format(out_file))

