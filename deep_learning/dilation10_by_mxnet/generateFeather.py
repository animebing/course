#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import argparse
import caffe
import cv2
import json
import numba
import numpy as np
from os.path import dirname, exists, join, splitext
import sys
import util

import mxnet as mx
__author__ = 'Bingbing'


def predict(dataset_name, model_prefix, epoch, dev, img_dir, output_dir):
    img_list = open(join(img_dir, 'valImage.lst'), 'r').readlines()

    symbol, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, epoch)
    input_shape = (1, 3, 1396, 2420)
    mean_pixel = np.array([72.39, 82.91, 73.16])
    exe = symbol.simple_bind(ctx=dev,data=input_shape)
    for name in exe.arg_dict:
        if name != 'data' and name != 'softmax_label':
            arg_params[name].copyto(exe.arg_dict[name])

    for name in exe.aux_dict:
        aux_params.copyto(exe.aux_dict[name])

    label_margin = 186

    batch_size, num_channels, input_height, input_width = input_shape
    caffe_in = np.zeros(input_shape, dtype=np.float32)
    for i in xrange(len(img_list)):
        img_name = join(img_dir, img_list[i]).strip()
        image = cv2.imread(img_name, 1).astype(np.float32) - mean_pixel
        image_size = image.shape
        output_height = input_height - 2 * label_margin
        output_width = input_width - 2 * label_margin
        image = cv2.copyMakeBorder(image, label_margin, label_margin,
                                    label_margin, label_margin,
                                    cv2.BORDER_REFLECT_101)
        num_tiles_h = image_size[0] // output_height + \
                    (1 if image_size[0] % output_height else 0)
        num_tiles_w = image_size[1] // output_width + \
                    (1 if image_size[1] % output_width else 0)
        prediction = []
        for h in range(num_tiles_h):
            col_prediction = []
            for w in range(num_tiles_w):
                offset = [output_height * h,
                        output_width * w]
                tile = image[offset[0]:offset[0] + input_height,
                            offset[1]:offset[1] + input_width, :]
                margin = [0, input_height - tile.shape[0],
                        0, input_width - tile.shape[1]]
                tile = cv2.copyMakeBorder(tile, margin[0], margin[1],
                                        margin[2], margin[3],
                                        cv2.BORDER_REFLECT_101)
                caffe_in[0] = tile.transpose([2, 0, 1]) # form H*W*3 to 3*H*W
                #caffe_in[:, [0, 2], :, :] = caffe_in[:, [2, 0], :, :] # from BGR to RGB
                exe.arg_dict['data'][0:1, :, :, :] = mx.nd.array(caffe_in, dev)
                out = exe.forward()
                prob = out[0].asnumpy()[0]
                col_prediction.append(prob)
            col_prediction = np.concatenate(col_prediction, axis=2)
            prediction.append(col_prediction)
        prob = np.concatenate(prediction, axis=1)

        prediction = np.argmax(prob.transpose([1, 2, 0]), axis=2)
        #color_image = dataset.palette[prediction.ravel()].reshape(image_size)
        #color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        img_name = img_list[i].split('/')[-1]
        output_path = join(output_dir, splitext(img_name)[0]+'_output.png')
        print('Writing', output_path)
        cv2.imwrite(output_path, prediction)

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
    parser.add_argument('output_dir', nargs='?', default='',
                        help='directory to output')
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

    predict(args.dataset, args.model_prefix, args.epoch, dev, args.img_dir, args.output_dir)


if __name__ == '__main__':
    main()
