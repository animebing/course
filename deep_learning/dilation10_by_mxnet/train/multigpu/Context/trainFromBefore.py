#`/uisr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import argparse
import mxnet as mx
import numpy as np
import logging
import os
#from data import FileIter
import symbol_sets as symSets
from metric import *
from initializer import Identity
import time
import memonger

#logger = logging.getLogger()
#logger.setLevel(logging.INFO)

def main():
    logging.basicConfig(format='%(asctime)s %(message)s', filename=os.path.join(args.work_dir, args.log_file), filemode='w', level=logging.INFO)
    t_start = time.time()
    """
    train_fea = mx.nd.load(os.path.join(args.data_dir, "trainFea.nd"))
    train_label = mx.nd.load(os.path.join(args.data_dir, "trainGt.nd"))

    train_iter = mx.io.NDArrayIter(train_fea, label=train_label, batch_size=args.batch_size, shuffle=True)

    if args.test:
        test_fea = mx.nd.load(os.path.join(args.data_dir, "valFea.nd"))
        test_label = mx.nd.load(os.path.join(args.data_dir, "valGt.nd"))
        test_iter = mx.io.NDArrayIter(test_fea, label=test_label, batch_size=args.batch_size, shuffle=True)
    else:
        test_iter = None

    """
    softmax, arg_params, aux_params = mx.model.load_checkpoint(args.params, args.params_epoch)
    print(aux_params.keys())
    #if args.model == "frontend":
    #    sym = symSets.front_end(args.classes)
    #elif args.model == "context":
    #    data = mx.sym.Variable('data')
    #    sym = symSets.dilation9(data, args.classes)
    #else:
    #    pass

    #sym = mx.sym.Deconvolution(data=sym, kernel=(2*8, 2*8), stride=(8, 8), pad=(8//2, 8//2), num_filter=args.classes, num_group=args.classes, attr={"weight_lr_mult":"0.0"}, name="ctx_upsample")

    #softmax = mx.sym.SoftmaxOutput(data=sym, multi_output=True, use_ignore=True, \
    #        ignore_label=255, name="softmax")
    #cost_1 = memonger.get_cost(softmax, data=(1, 19, 64, 128))
    #print(cost_1)
    #_, arg_params, aux_params = mx.model.load_checkpoint(args.params, args.params_epoch)
    #arg_params = {}
    #arg_names = softmax.list_arguments()
    #print(train_iter.provide_data)
    #arg_shapes, _, _ = softmax.infer_shape(data=train_iter.provide_data[0].shape)
    #arg_shape_dict = dict(zip(arg_names, arg_shapes))
    #arg_params['ctx_upsample_weight'] = np.zeros(arg_shape_dict['ctx_upsample_weight'])
    #mx.init.Initializer()._init_bilinear('ctx_upsample_weight', arg_params['ctx_upsample_weight'])

    """
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    epoch_size = train_iter.num_data//train_iter.batch_size
    lr_scheduler = mx.lr_scheduler.FactorScheduler(step = max(int(epoch_size * args.lr_factor_epoch), 1), factor = args.lr_factor)

    model = mx.model.FeedForward(ctx=ctx, symbol=softmax, num_epoch=args.epoches, optimizer='sgd', arg_params=arg_params, begin_epoch=args.params_epoch, learning_rate=args.lr, momentum=args.momentum, wd=args.wd, lr_scheduler=lr_scheduler)

    eval_metric = [ClassAccuracy(), ClassCrossEntropy()]

    model.fit(X=train_iter, eval_data=test_iter, eval_metric=eval_metric, batch_end_callback=mx.callback.Speedometer(args.batch_size, args.batch_interval), epoch_end_callback=mx.callback.do_checkpoint(os.path.join(args.work_dir, args.model)))


    t_end = time.time()
    m, s = divmod(t_end-t_start, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    logging.info("Time: %d days, %d hours, %d minutes, %d seconds", d, h, m, s)
    """
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs='?',
                        choices=["frontend", "context", "joint"])
    parser.add_argument("--params", default=None,
                        help="path to the params to initialize the model")
    parser.add_argument("--params_epoch", type=int, default=None,
                        help="the epoch for the loaded params")
    parser.add_argument("--mean", nargs='*', type=float, default=[72.39, 82.91, 73.16],
                        help="mean pixel vaule(BGR) for the dataset.\n"
                        "default is the mean pixel of PASCAL dataset.")
    parser.add_argument("--work_dir", default=".",
                        help="working dir for trarining. \nall the generated "
                             "network and solver configurations will be "
                             "written to this directory, in addition to "
                             "training snapshots")
    parser.add_argument('--log_file', default="frontend.log",
                        help="the file to store log")
    parser.add_argument("--data_dir", default="", required=True,
                        help="data dir")
    parser.add_argument("--train", action="store_true",
                        help="path to the training list")
    parser.add_argument("--test", action="store_true",
                        help="path to the testing list")
    parser.add_argument("--batch_size", type=int, default=6,
                        help="batch size")
    parser.add_argument("--batch_interval", type=int, default=10,
                        help="the interval for batch_end_callback")
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
    parser.add_argument("--std", type=float, default=0.01,
                        help="std for gaussian in initializer")
    args = parser.parse_args()
    main()



