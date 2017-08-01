#!/uisr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import argparse
import mxnet as mx
import numpy as np
import logging
import os
from data import FileIter
import symbol_sets as symSets
from metric import *
from initializer import Identity
import time
import memonger

#logger = logging.getLogger()
#logger.setLevel(logging.INFO)


def main():
    t_start = time.time()
    val_iter = FileIter(root_dir=args.data_dir, flist_name=args.test_list, up=args.up,
                        rgb_mean=tuple(args.mean), batch_size=args.test_batch)
    sym = symSets.front_end(args.classes)
    sym = symSets.dilation10(sym, args.classes)

    if args.up:
        sym = mx.sym.Deconvolution(data=sym, kernel=(2*8, 2*8), stride=(8, 8), pad=(8//2, 8//2), num_filter=args.classes, num_group=args.classes, attr={"weight_lr_mult":"0.0"}, name="ctx_upsample")

    softmax= mx.sym.SoftmaxOutput(data=sym, multi_output=True, use_ignore=True, \
            ignore_label=255, name="softmax")
    """
    cost_1 = memonger.get_cost(softmax, data=(1, 3, 1396, 1396))
    print("feature cost: ", cost_1)
    """

    _, arg_params, aux_params = mx.model.load_checkpoint(args.params, args.params_epoch)

    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]

    model = mx.model.FeedForward(ctx=ctx, symbol=softmax, initializer=Identity(args.std), arg_params=arg_params, aux_params=aux_params)

    eval_metric = [ClassAccuracy(), ClassCrossEntropy()]
    model._init_params(dict(val_iter.provide_data+val_iter.provide_label))
    values = model.score(val_iter, eval_metric=eval_metric, reset=True)
    print(values)

    t_end = time.time()
    m, s = divmod(t_end-t_start, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    print("Time: %d days, %d hours, %d minutes, %d seconds" %(d, h, m, s))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str,
                        help="path to the params to initialize the model")
    parser.add_argument("--params_epoch", type=int,
                        help="the epoch for the loaded params")
    parser.add_argument("--mean", nargs='*', type=float, default=[72.39, 82.91, 73.16],
                        help="mean pixel vaule(BGR) for the dataset.\n"
                        "default is the mean pixel of PASCAL dataset.")
    parser.add_argument("--data_dir", default="", required=True,
                        help="data dir")
    parser.add_argument("--test_list", default="",
                        help="path to the testing list")
    parser.add_argument("--test_batch", type=int, default=2,
                        help="testing batch size. if 0, no test phase")
    parser.add_argument("--classes", type=int, required=True,
                        help="number of class in the data")
    parser.add_argument("--gpus", type=str, default="0",
                        help="GPU index for training")
    parser.add_argument("--std", type=float,
                        help="standard deviation for gaussian")
    parser.add_argument("--up", action="store_true", default=True,
                        help="if true, upsampling the final feature map "
                            "before calculating the loss or accuracy")

    args = parser.parse_args()
    main()



