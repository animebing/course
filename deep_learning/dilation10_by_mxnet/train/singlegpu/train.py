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
from solver import Solver
from metric import *
import time


#logger = logging.getLogger()
#logger.setLevel(logging.INFO)


def main():
    logging.basicConfig(filename=os.path.join(args.work_dir, args.log_file), filemode='w', level=logging.INFO)
    t_start = time.time()
    train_iter = FileIter(root_dir=args.data_dir, flist_name=args.train_list, up=args.up,
                         crop_size=args.crop_size, rgb_mean=tuple(args.mean),
                         batch_size=args.train_batch)
    if args.test_list != "":
        val_iter = FileIter(root_dir=args.data_dir, flist_name=args.test_list, up=args.up,
                            crop_size=args.crop_size, rgb_mean=tuple(args.mean),
                            batch_size=args.test_batch)
    else:
        val_iter = None
    if args.model == "frontend":
        sym = symSets.front_end(args.classes)
    elif args.model == "context":
        pass
    else:
        pass

    if args.up:
        sym = mx.sym.Deconvolution(data=sym, kernel=(2*8, 2*8), stride=(8, 8), pad=(8//2, 8//2), num_filter=args.classes, num_group=args.classes, name="ctx_upsample")

    softmax= mx.sym.SoftmaxOutput(data=sym, multi_output=True, use_ignore=True, \
            ignore_label=255, name="softmax")
    arg_names = softmax.list_arguments()
    aux_names = softmax.list_auxiliary_states()
    ctx = mx.gpu(args.gpu)
    _, arg_tmp, aux_tmp = mx.model.load_checkpoint(args.params, args.params_epoch)
    arg_params = {}
    aux_params = {}
    for name in arg_tmp:
        if name in arg_names:
            arg_params[name] = mx.nd.zeros(arg_tmp[name].shape, ctx)
            arg_tmp[name].copyto(arg_params[name])

    for name in aux_tmp:
        if name in aux_names:
            aux_params[name] = mx.nd.zeros(aux_tmp[name].shape, ctx)
            aux_tmp[name].copyto(aux_params[name])

    input_shape = train_iter.provide_data[0][1]
    arg_shapes, _, _ = softmax.infer_shape(data=input_shape)
    arg_names = softmax.list_arguments()
    arg_shape_dict = dict(zip(arg_names, arg_shapes))
    arg_params["final_weight"] = mx.nd.zeros(arg_shape_dict["final_weight"], ctx)
    arg_params["final_bias"] = mx.nd.zeros(arg_shape_dict["final_bias"], ctx)
    mx.init.Normal(0.001)._init_weight("final_weight", arg_params["final_weight"])
    mx.init.Initializer()._init_zero("final_bias", arg_params["final_bias"])
    # how to deal with the extra params which are not appearing in my symbol
    if args.up:
        arg_params['ctx_upsample_weight'] = mx.nd.zeros(arg_shape_dict['ctx_upsample_weight'], ctx)
        init = mx.init.Initializer()
        init._init_bilinear('ctx_upsample', arg_params['ctx_upsample_weight'])
    model = Solver(ctx = ctx, symbol = softmax, begin_epoch = 0,
                  num_epoch = 5, arg_params=arg_params, aux_params=aux_params,
                  lr_factor=args.lr_factor, learning_rate=args.lr, momentum=args.momentum, wd=args.wd)
    eval_metric = [ClassAccuracy(), ClassCrossEntropy()]
    model.fit(train_data=train_iter, eval_data=val_iter, eval_metric=eval_metric,
             batch_end_callback=mx.callback.Speedometer(args.train_batch, 10),
             epoch_end_callback=mx.callback.do_checkpoint(os.path.join(args.work_dir, args.model)))

    t_end = time.time()
    m, s = divmod(t_end-t_start, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    logging.info("Time: %d days, %d hours, %d minutes, %d seconds", d, h, m, s)

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
    parser.add_argument("--train_batch", type=int, default=2,
                        help="training batch size")
    parser.add_argument("--test_batch", type=int, default=2,
                        help="testing batch size. if 0, no test phase")
    parser.add_argument("--crop_size", type=int, default=628)
    parser.add_argument("--lr", type=float, default=0.0,
                        help="solver SGD learning rate")
    parser.add_argument("--lr_factor", type=float, default=1.0,
                        help="learning rate epoch factor")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="gradient momentum")
    parser.add_argument("--wd", type=float, default=0.0,
                        help="weight decay")
    parser.add_argument("--classes", type=int, required=True,
                        help="number of class in the data")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index for training")
    parser.add_argument("--up", action="store_true", default=True,
                        help="if true, upsampling the final feature map "
                            "before calculating the loss or accuracy")
    parser.add_argument("--layers", type=int, default=8,
                        help="used fo training context module.\n"
                            "number of layers in the context module.")

    args = parser.parse_args()
    #print(args.mean)
    main()



