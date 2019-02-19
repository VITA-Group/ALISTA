#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
file  : data.py
author: Xiaohan Chen
email : chernxh@tamu.edu
last_modified: 2018-10-16

Utility methods for the real world images compressive sensing experiments.
"""

import os
import glob
import numpy as np
import tensorflow as tf

from PIL import Image
from tqdm import tqdm
from utils.prob import load_problem
from sklearn.feature_extraction.image import extract_patches_2d
tqdm.monitor_interval = 0

def _bytes_feature(value):
    return tf.train.Feature (bytes_list=tf.train.BytesList (value=[value]))


def dir2tfrecords (data_dir, out_path, Phi, patch_size, patches_per_image):
    Phi = Phi.astype (np.float32)
    if isinstance (patch_size, int):
        patch_size = (16,16)

    writer = tf.python_io.TFRecordWriter (out_path)
    for fn in tqdm (glob.glob (os.path.join (data_dir, "*.jpg"))) :
        """Read images (and convert to grayscale)."""
        im = Image.open (fn)
        if im.mode == 'RGB':
            im = im.convert ('L')
        im = np.asarray (im)

        """Extract patches."""
        patches = extract_patches_2d (im, patch_size)
        perm = np.random.permutation (len (patches))
        patches = patches [perm [:patches_per_image]]

        """Vectorize patches."""
        fs = patches.reshape (len (patches), -1)

        """Demean and normalize."""
        fs = fs -  np.mean (fs, axis=1, keepdims=True)
        fs = (fs / 255.0).astype (np.float32)

        """Measure the signal using sensing matrix `Phi`."""
        ys = np.transpose (Phi.dot (np.transpose (fs)))

        """Write singals and measurements to tfrecords file."""
        for y, f in zip (ys, fs):
            yraw = y.tostring ()
            fraw = f.tostring ()
            example = tf.train.Example (features=tf.train.Features (
                feature={
                    'y': _bytes_feature (yraw),
                    'f': _bytes_feature (fraw)
                }
            ))

            writer.write (example.SerializeToString ())

    writer.close ()


"""*****************************************************************************
Input pipeline for real images compressive sensing on preprocessed BSD500
datasets.
*****************************************************************************"""
def decode (serialized_example):
    """Parses an image from the given `serialized_example`."""
    features = tf.parse_single_example (
        serialized_example,
        features={
            'y' : tf.FixedLenFeature ([], tf.string),
            'f' : tf.FixedLenFeature ([], tf.string),
        })

    # convert from a scalar string tensor to a uint8 tensor with
    # shape (heigth, width, depth)
    y_ = tf.decode_raw (features ['y'], tf.float32)
    f_ = tf.decode_raw (features ['f'], tf.float32)

    return y_, f_


def bsd500_cs_inputs (file_path, batch_size, num_epochs):
    if not num_epochs:
        num_epochs = None

    with tf.name_scope ('input'):
        # TFRecordDataset opens a binary file and reads one record at a time.
        # `filename` could also be a list of filenames, which will be read in order.
        dataset = tf.data.TFRecordDataset (file_path)

        # The map transformation takes a function and applies it to every element of
        # the dataset
        dataset = dataset.map (decode, num_parallel_calls=4)

        # The shuffle transformation uses a finite-sized buffer to shuffle elements
        # in memory. The parameter is the number of elements in the buffer. For
        # completely uniform shuffling, set the parameter to be the same as the
        # number of elements in the dataset.
        dataset = dataset.apply (
                tf.contrib.data.shuffle_and_repeat (50000, num_epochs))

        dataset = dataset.batch (batch_size)
        dataset = dataset.prefetch (batch_size)

        iterator = dataset.make_one_shot_iterator ()

        # After this step, the y_ and f_ will be of shape:
        # (batch_size, M) and (batch_size, F). Transpose them into shape
        # (M, batch_size) and (F, batch_size).
        y_, f_ = iterator.get_next ()

    return tf.transpose (y_, [1,0]) , tf.transpose (f_, [1,0])

