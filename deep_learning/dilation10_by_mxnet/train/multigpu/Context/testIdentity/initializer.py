# coding: utf-8

from __future__ import print_function, division
import mxnet as mx
import numpy as np

class Identity(mx.initializer.Initializer):
    """Initialize the weight so that the output is equal to input

    Parameters
    ---------
    std: standart deviation for guassian weight

    """
    def __init__(self, std=0.01):
        self.std = std

    def _init_weight(self, name, arr):
        if name.endswith("weight"):
            n, c, h, w = arr.shape
            h_idx = h//2
            w_idx = w//2
            if self.std > 0:
                tmp = self.std*np.random.randn(n, c, h, w)
            else:
                tmp = np.zeros((n, c, h, w))
            for i in xrange(n):
                tmp[i, i, h_idx, w_idx] = 1

            arr[:] = mx.nd.array(tmp)
