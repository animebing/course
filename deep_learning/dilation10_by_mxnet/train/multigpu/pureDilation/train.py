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
import time
import memonger

def main():

    logging.basicConfig(format='%(asctime)s %(message)s', filename=os.path.join(args.work_dir, args.log_file), filemode='w', level=logging.INFO)
    t_start = time.time()
    train_iter = FileIter(root_dir=args.data_dir, flist_name=args.train_list,
                         crop_size=args.crop_size, rgb_mean=tuple(args.mean),
                         batch_size=args.train_batch)
    if args.test_list != "":
        val_iter = FileIter(root_dir=args.data_dir, flist_name=args.test_list,
                            crop_size=args.crop_size, rgb_mean=tuple(args.mean),
                            batch_size=args.test_batch)
    else:
        val_iter = None


    data = mx.sym.Variable("data")
    sym = symSets.dilation10(data, args.classes)

    softmax= mx.sym.SoftmaxOutput(data=sym, multi_output=True, use_ignore=True, \
            ignore_label=255, name="softmax")

    #cost = memonger.get_cost(softmax, data=(1, 3, 1024, 1024))

    #print("cost: ", cost)

    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    lr_scheduler = mx.lr_scheduler.FactorScheduler(step = max(int(train_iter.max_iter * args.lr_factor_epoch), 1), factor = args.lr_factor)

    model = mx.model.FeedForward(ctx=ctx, symbol=softmax, num_epoch=args.epoches, optimizer='sgd', initializer=mx.init.Normal(), begin_epoch=0, learning_rate=args.lr, momentum=args.momentum, wd=args.wd, lr_scheduler=lr_scheduler)

    eval_metric = [ClassAccuracy(), ClassCrossEntropy()]

    model.fit(X=train_iter, eval_data=val_iter, eval_metric=eval_metric, batch_end_callback=mx.callback.Speedometer(args.train_batch, args.batch_interval), epoch_end_callback=mx.callback.do_checkpoint(os.path.join(args.work_dir, args.model)))


    t_end = time.time()
    m, s = divmod(t_end-t_start, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    logging.info("Time: %d days, %d hours, %d minutes, %d seconds", d, h, m, s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs='?',
                        choices=["frontend", "context", "joint", "pure"])
    parser.add_argument("--mean", nargs='*', type=float, default=[72.39, 82.91, 73.16],
                        help="mean pixel vaule(BGR) for the dataset.\n"
                        "default is the mean pixel of PASCAL dataset.")
    parser.add_argument("--work_dir", default="model/",
                        help="working dir for trarining. \nall the generated "
                             "network and solver configurations will be "
                             "written to this directory, in addition to "
                             "training snapshots")
    parser.add_argument('--log_file', default="frontend.log",
                        help="the file to store log")
    parser.add_argument("--data_dir", default="", required=True,
                        help="data dir")
    parser.add_argument("--train_list", default="", required=True,
                        help="path to the training list")
    parser.add_argument("--test_list", default="",
                        help="path to the testing list")
    parser.add_argument("--train_batch", type=int, default=6,
                        help="training batch size")
    parser.add_argument("--test_batch", type=int, default=6,
                        help="testing batch size. if 0, no test phase")
    parser.add_argument("--batch_interval", type=int, default=10,
                        help="the interval for batch_end_callback")
    parser.add_argument("--crop_size", type=int, default=628,
                        help="crop size for training")
    parser.add_argument("--epoches", type=int, default=1,
                        help="the epoches for training")
    parser.add_argument("--lr", type=float, default=0.0,
                        help="solver SGD learning rate")
    parser.add_argument("--lr_factor", type=float, default=1.0,
                        help="lr_factor for lr_scheduler")
    parser.add_argument("--lr_factor_epoch", type=int, default=1,
                        help="epoch for lr_factor")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="gradient momentum")
    parser.add_argument("--wd", type=float, default=0.0,
                        help="weight decay")
    parser.add_argument("--classes", type=int, required=True,
                        help="number of class in the data")
    parser.add_argument("--gpus", type=str, default="0",
                        help="GPU index for training")

    args = parser.parse_args()
    #print(args.mean)
    main()



