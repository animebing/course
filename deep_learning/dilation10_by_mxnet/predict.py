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
#import cityscapes_symbol as csym
__author__ = 'Fisher Yu'
__copyright__ = 'Copyright (c) 2016, Fisher Yu'
__email__ = 'i@yf.io'
__license__ = 'MIT'


class Dataset(object):
    def __init__(self, dataset_name):
        self.work_dir = dirname(__file__)
        info_path = join(self.work_dir, 'datasets', dataset_name + '.json')
        if not exists(info_path):
            raise IOError("Do not have information for dataset {}"
                          .format(dataset_name))
        with open(info_path, 'r') as fp:
            info = json.load(fp)
        self.input_shape = tuple(info['input_shape'])

        self.palette = np.array(info['palette'], dtype=np.uint8)
        self.mean_pixel = np.array(info['mean'], dtype=np.float32)
        self.dilation = info['dilation']
        self.zoom = info['zoom']
        self.name = dataset_name
        self.model_name = 'dilation{}_{}'.format(self.dilation, self.name)
        self.model_path = join(self.work_dir, 'models',
                               self.model_name + '_deploy.prototxt')

    @property
    def pretrained_path(self):
        p = join(dirname(__file__), 'pretrained',
                 self.model_name + '.caffemodel')
        if not exists(p):
            download_path = join(self.work_dir, 'pretrained',
                                 'download_{}.sh'.format(self.name))
            raise IOError('Pleaes run {} to download the pretrained network '
                          'weights first'.format(download_path))
        return p


def predict(dataset_name, epoch, dev, input_path, output_path):
    dataset = Dataset(dataset_name)

    model = mx.model.FeedForward.load(dataset_name, epoch, dev)
    symbol, arg_params, aux_params = mx.model.load_checkpoint(dataset_name, epoch)
    exe = symbol.simple_bind(ctx=dev,data=tuple(dataset.input_shape))
    for name in exe.arg_dict:
        if name != 'data' and name != 'softmax_label':
            arg_params[name].copyto(exe.arg_dict[name])

    for name in exe.aux_dict:
        aux_params.copyto(exe.aux_dict[name])


    """
    if dataset_name == "cityscapes":
        sym = model.symbol
        arg_list = sym.list_arguments()
        arg_shapes, _, _ = sym.infer_shape(data=dataset.input_shape)
        arg_shape_dict = dict(zip(arg_list, arg_shapes))
        model.arg_params['ctx_upsample_weight'] = mx.nd.zeros(arg_shape_dict['ctx_upsample_weight'])
        init = mx.init.Initializer()
        init._init_bilinear('ctx_upsample', model.arg_params['ctx_upsample_weight'])
    #print(len(model.arg_params))
    #print(model.arg_params.keys())
    """
    label_margin = 186
    input_shape = dataset.input_shape
    model.arg_params['data'] = mx.nd.zeros(tuple(input_shape), dev)
    batch_size, num_channels, input_height, input_width = input_shape
    caffe_in = np.zeros(input_shape, dtype=np.float32)
    image = cv2.imread(input_path, 1).astype(np.float32) - dataset.mean_pixel

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
            #out = model.predict(caffe_in)
            exe.arg_dict['data'][0:1, :, :, :] = mx.nd.array(caffe_in, dev)
            out = exe.forward()
            prob = out[0].asnumpy()[0]
            col_prediction.append(prob)
        # print('concat row')
        col_prediction = np.concatenate(col_prediction, axis=2)
        prediction.append(col_prediction)
    prob = np.concatenate(prediction, axis=1)
    print("prediction shape: ", prob.shape[1]//num_tiles_h, prob.shape[2]//num_tiles_w)
    print("concatenated shape: ", prob.shape)
    if dataset.zoom > 1:
        prob = util.interp_map(prob, dataset.zoom, image_size[1], image_size[0])
    prediction = np.argmax(prob.transpose([1, 2, 0]), axis=2)
    #print(prediction.shape)
    #iimage_size = (128, 256, 3)
    # get the color for each class
    color_image = dataset.palette[prediction.ravel()].reshape(image_size)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    print('Writing', output_path)
    cv2.imwrite(output_path, color_image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', nargs='?')
                       # choices=['pascal_voc', 'camvid', 'kitti', 'cityscapes'])
    parser.add_argument('epoch', type=int, default=1,
                        help='the epoch of loaded model parameters')
    parser.add_argument('input_path', nargs='?', default='',
                        help='Required path to input image')
    parser.add_argument('-o', '--output_path', default=None)

    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU ID to run CAFFE. '
                             'If -1 (default), CPU is used')

    args = parser.parse_args()

    if args.input_path == '':
        raise IOError('Error: No path to input image')
    if not exists(args.input_path):
        raise IOError("Error: Can't find input image " + args.input_path)
    if args.gpu >= 0:
        dev = mx.gpu(args.gpu)
        print('Using GPU ', args.gpu)
    else:
        dev = mx.cpu()
        print('Using CPU')

    if args.output_path is None:
        args.output_path = '{}_{}.png'.format(
                splitext(args.input_path)[0], 'output')
    predict(args.dataset, args.epoch, dev, args.input_path, args.output_path)


if __name__ == '__main__':
    main()
