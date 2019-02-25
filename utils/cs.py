#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
file  : cs.py
author: Xiaohan Chen
email : chernxh@tamu.edu
date  : 2019-02-18

Utility functions for natural images compressive sensing.
"""

import numpy as np
from PIL import Image

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

