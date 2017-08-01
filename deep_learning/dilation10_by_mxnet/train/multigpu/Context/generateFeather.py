#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import argparse
import cv2
import json
import numpy as np
from os.path import dirname, exists, join, splitext
import sys
from symbol_sets import *
import memonger
import mxnet as mx
__author__ = 'Bingbing'


def predict(dataset_name, model_prefix, epoch, dev, img_dir, category):
    img_list = category+"Image.lst"
    img_list = open(join(img_dir, img_list), 'r').readlines()
    label_list = category+"Gt.lst"
    label_list = open(join(img_dir, label_list), 'r').readlines()

    _, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, epoch)
    symbol = front_end()
    input_shape = (1, 3, 884, 1396)  # (1, 3, 1024//2+186*2, 2048//2+186*2)
    #cost = memonger.get_cost(symbol, data=input_shape)
    #print(cost)

    mean_pixel = np.array([72.39, 82.91, 73.16])
    exe = symbol.simple_bind(ctx=dev, data=input_shape)
    for name in exe.arg_dict:
        if name != 'data':
            arg_params[name].copyto(exe.arg_dict[name])


    label_margin = 186

    batch_size, num_channels, input_height, input_width = input_shape
    caffe_in = np.zeros(input_shape, dtype=np.float32)
    features = []
    labels = []
    for i in xrange(len(img_list)):
        img_name = img_list[i].strip()
        img = cv2.imread(img_name, 1).astype(np.float32)
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        img = img-mean_pixel
        img = cv2.copyMakeBorder(img, label_margin, label_margin,
                                    label_margin, label_margin,
                                    cv2.BORDER_REFLECT_101)

        label_name = label_list[i].strip()
        label = cv2.imread(label_name, 0)
        label = cv2.resize(label, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

        caffe_in[0] = img.transpose([2, 0, 1]) # form H*W*3 to 3*H*W
        exe.arg_dict['data'][0:1, :, :, :] = mx.nd.array(caffe_in, dev)
        out = exe.forward()

        features.append(out[0].asnumpy())
        labels.append(label.ravel().reshape(1, -1))
        print(img_name)
    """
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    img_nd = category+"Fea.nd"
    mx.nd.save(join(img_dir, img_nd), [mx.nd.array(features)])
    label_nd = category+"Gt.nd"
    mx.nd.save(join(img_dir, label_nd), [mx.nd.array(labels)])
    """
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', nargs='?')
                       # choices=['pascal_voc', 'camvid', 'kitti', 'cityscapes'])
    parser.add_argument('model_prefix', type=str, default="frontend",
                        help="the model prefix")
    parser.add_argument('epoch', type=int, default=1,
                        help='the epoch of loaded model parameters')
    parser.add_argument('img_dir', nargs='?', default='',
                        help='directory to input images')
    parser.add_argument('category', type=str, default="train",
                        help='the image category to be processed')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU ID to run CAFFE. '
                             'If -1 (default), CPU is used')

    args = parser.parse_args()

    if args.img_dir == '':
        raise IOError('Error: no directory to images')
    if args.gpu >= 0:
        dev = mx.gpu(args.gpu)
        print('Using GPU ', args.gpu)
    else:
        dev = mx.cpu()
        print('Using CPU')

    predict(args.dataset, args.model_prefix, args.epoch, dev, args.img_dir, args.category)


if __name__ == '__main__':
    main()
