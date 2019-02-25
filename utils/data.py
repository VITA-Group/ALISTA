#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
file  : data.py
author: Xiaohan Chen
email : chernxh@tamu.edu
date  : 2019-02-18

Utility methods for data handling for natural images denoising and compressive
sensing experiments.
"""

import os
import sys
import glob
import argparse
import numpy as np
import tensorflow as tf

sys.path.append(os.getcwd())

from PIL import Image
from tqdm import tqdm
from utils.prob import load_problem
from sklearn.feature_extraction.image import extract_patches_2d
tqdm.monitor_interval = 0

def _int64_feature(value):
  return tf.train.Feature (int64_list=tf.train.Int64List (value=[value]))

def _bytes_feature(value):
    return tf.train.Feature (bytes_list=tf.train.BytesList (value=[value]))

def dir2tfrecords_cs (data_dir, out_path, Phi, patch_size, patches_per_image, suffix):
    Phi = Phi.astype (np.float32)
    if isinstance (patch_size, int):
        patch_size = (16,16)

    if not out_path.endswith(".tfrecords"):
        out_path += ".tfrecords"
    writer = tf.python_io.TFRecordWriter (out_path)
    for fn in tqdm (glob.glob (os.path.join (data_dir, "*." + suffix))) :
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


def dir2tfrecords_denoise (data_dir, out_dir, name, suffix, rgb2gray=False):
    images = glob.glob (os.path.join (data_dir, '*.' + suffix))
    num_examples = len (images)

    if len (images) < 1:
        raise ValueError ('No images in {suffix} format found in {data_dir}'.format (suffix, data_dir))

    if not name.endswith(".tfrecords"):
        name += ".tfrecords"
    out_fname = os.path.join(out_dir, name)
    print ('Writing', out_fname)
    with tf.python_io.TFRecordWriter(out_fname) as writer:
        for fname in images:
            image = Image.open(fname)
            width, height = image.size
            if image.mode == 'L':
                depth = 1
            elif image.mode == 'RGB':
                if rgb2gray:
                    image = image.convert('L')
                    depth = 1
                else:
                    depth = 3
            image_raw = np.array(image).tostring()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'height': _int64_feature(height),
                        'width' : _int64_feature(width),
                        'depth' : _int64_feature(depth),
                        'image_raw': _bytes_feature(image_raw)
                    }))
            writer.write (example.SerializeToString ())


"""*****************************************************************************
Input pipeline for real images compressive sensing on preprocessed BSD500
datasets.
*****************************************************************************"""
def cs_decode(serialized_example):
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
        dataset = dataset.map (cs_decode, num_parallel_calls=4)

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


"""*****************************************************************************
Input pipeline for real images denoising on BSD500 and VOC2012 datasets.
*****************************************************************************"""
def denoise_decode (serialized_example):
    """Parses an image from the given `serialized_example`."""
    features = tf.parse_single_example (
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature ([], tf.string),
            'height': tf.FixedLenFeature ([], tf.int64),
            'width' : tf.FixedLenFeature ([], tf.int64),
            'depth' : tf.FixedLenFeature ([], tf.int64),
        })

    # convert from a scalar string tensor to a uint8 tensor with
    # shape (heigth, width, depth)
    image = tf.decode_raw (features ['image_raw'], tf.uint8)

    height = tf.cast (features ['height'], tf.int32)
    width  = tf.cast (features ['width'],  tf.int32)
    depth  = tf.cast (features ['depth'],  tf.int32)
    image  = tf.cast (tf.reshape (image, (height, width, depth)), tf.float32)

    return image

def normalization (image):
    """Convert `image` from [0, 255] -> [0, 1] floats and then de-mean."""
    image = image * (1. / 255)
    return image - tf.reduce_mean (image)

def crop (image, height_crop, width_crop):
    """Randomly crop images to size (height_crop, width_crop)."""
    image = tf.random_crop (image, [height_crop, width_crop, 1])
    return image

def bsd500_denoise_inputs (dataset_dir, filename, batch_size,
                           height_crop, width_crop, num_epochs):
    if not num_epochs:
        num_epochs = None
    filename = os.path.join (dataset_dir, filename)

    with tf.name_scope ('input'):
        # TFRecordDataset opens a binary file and reads one record at a time.
        # `filename` could also be a list of filenames, which will be read in order.
        dataset = tf.data.TFRecordDataset (filename)

        # The map transformation takes a function and applies it to every element of
        # the dataset
        dataset = dataset.map(denoise_decode, num_parallel_calls=4)
        dataset = dataset.map(lambda x: crop(x, height_crop, width_crop),
                              num_parallel_calls=4)
        dataset = dataset.map(normalization, num_parallel_calls=4)

        # The shuffle transformation uses a finite-sized buffer to shuffle elements
        # in memory. The parameter is the number of elements in the buffer. For
        # completely uniform shuffling, set the parameter to be the same as the
        # number of elements in the dataset.
        dataset = dataset.apply (
                tf.data.experimental.shuffle_and_repeat (400, num_epochs))

        dataset = dataset.batch (batch_size)
        dataset = dataset.prefetch (batch_size)

        iterator = dataset.make_one_shot_iterator ()

        image_ = iterator.get_next ()

    return image_


parser = argparse.ArgumentParser()
parser.add_argument(
    "--task_type", type=str, help="Denoise `denoise` or compressive sensing `cs`.")
parser.add_argument(
    "--dataset_dir", type=str, help="Path to the directory that holds images.")
parser.add_argument(
    "--out_dir", type=str, help="Path to the output directory that holds the TFRecords file.")
parser.add_argument(
    "--out_file", type=str, help="File name of the output file.")
parser.add_argument(
    "--suffix", type=str, help="Format of input images. PNG or JPG or other format.")
# Arguments for compressive sensing
parser.add_argument(
    "--sensing", type=str, help="Sensing matrix file. Instance of Problem class.")
parser.add_argument(
    "--patch_size", type=int, help="Size of extracted patches.")
parser.add_argument(
    "--patches_per_img", type=int, help="How many patches to be extracted from each image.")

if __name__ == "__main__":
    config, unparsed = parser.parse_known_args()
    if config.task_type == "cs":
        Phi = np.load(config.sensing)["A"]
        dir2tfrecords_cs(config.dataset_dir,
                         os.path.join(config.out_dir, config.out_file),
                         config.patch_size, config.patches_per_img, config.img_fmt)
    elif config.task_type == "denoise":
        dir2tfrecords_denoise(config.dataset_dir, config.out_dir, config.out_file,
                              config.suffix, rgb2gray=True)

