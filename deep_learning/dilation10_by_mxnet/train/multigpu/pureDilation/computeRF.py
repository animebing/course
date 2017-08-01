#!/usr/bin/python
from __future__ import print_function, division
import argparse

conv_vgg = [[3, 1, 0], [3, 1, 0], [2, 2, 0], [3, 1, 0], [3, 1, 0], [2, 2, 0],
            [3, 1, 0], [3, 1, 0], [3, 1, 0], [2, 2, 0],
            [3, 1, 0], [3, 1, 0], [3, 1, 0], [2, 2, 0],
            [3, 1, 0], [3, 1, 0], [3, 1, 0]]

conv_vgg1 = [[3, 1, 0], [3, 1, 0], [2, 2, 0], [3, 1, 0], [3, 1, 0], [2, 2, 0],
            [3, 1, 0], [3, 1, 0], [3, 1, 0], [2, 2, 0],
            [3, 1, 0], [3, 1, 0], [3, 1, 0], [2, 2, 0],
            [3, 1, 0], [3, 1, 0], [3, 1, 0], [2, 2, 0],
            [7, 1, 0], [1, 1, 0], [1, 1, 0]]

conv_vgg2 = [[3, 1, 0], [3, 1, 0], [2, 2, 0], [3, 1, 0], [3, 1, 0], [2, 2, 0],
            [3, 1, 0], [3, 1, 0], [3, 1, 0], [2, 2, 0],
            [3, 1, 0], [3, 1, 0], [3, 1, 0],
            [5, 1, 0], [5, 1, 0], [5, 1, 0],
            [25, 1, 0], [1, 1, 0], [1, 1, 0]]

layer_name_vgg1 = ["conv1_1", "conv1_2", "pool1", "conv2_1", "conv2_2", "pool2",
                 "conv3_1", "conv3_2", "conv3_3", "pool3",
                 "conv4_1", "conv4_2", "conv4_3", "pool4",
                 "conv5_1", "conv5_2", "conv5_3", "pool5",
                 "fc6", "fc7", "final"]

layer_name_vgg2 = ["conv1_1", "conv1_2", "pool1", "conv2_1", "conv2_2", "pool2",
                 "conv3_1", "conv3_2", "conv3_3", "pool3",
                 "conv4_1", "conv4_2", "conv4_3",
                 "conv5_1", "conv5_2", "conv5_3",
                 "fc6", "fc7", "final"]

convnet= [[3, 1, 1], [3, 1, 1], [5, 1, 2], [9, 1, 4], [17, 1, 8], \
        [33, 1, 16], [65, 1, 32], [129, 1, 64], \
        [3, 1, 1], [1, 1, 0]]
conv_joint = conv_vgg2 + convnet
layer_name = ["ctx1_1", "ctx1_2", "ctx2_1", "ctx3_1", "ctx4_1", \
        "ctx5_1", "ctx6_1", "ctx7_1", \
        "ctx_fc1", "ctx_final"]
layer_name_joint = layer_name_vgg2 + layer_name

def outFromIn(img_size, lay_num=9, net=convnet):
    if lay_num > len(convnet): lay_num = len(convnet)

    total_stride = 1
    in_size = img_size

    for layer in xrange(lay_num):
        fsize, stride, pad = net[layer]
        out_size = (in_size+2*pad-fsize)//stride + 1
        in_size = out_size
        total_stride *= stride

    return out_size, total_stride




def inFromOut(layer_num, net = convnet):
    if layer_num > len(net): layer_num = len(net)
    in_size = 1

    for layer in reversed(range(layer_num)):
        fsize, stride, _ = net[layer]
        in_size = (in_size-1)*stride + fsize

    return in_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, nargs="*", required=True,
                        help="input image size")
    args = parser.parse_args()
    img_size = args.img_size
    out_r = []
    out_c = []
    rf = []
    for i in range(len(conv_joint)):
        #r_tmp, _ = outFromIn(img_size[0], i+1)
        #out_r.append(r_tmp)
        #c_tmp, _ = outFromIn(img_size[0], i+1)
        #out_c.append(c_tmp)

        rf_tmp = inFromOut(i+1, conv_joint)
        rf.append(rf_tmp)

    for i in xrange(len(conv_joint)):
        print("layer name = %s, RF size = %3d" %(layer_name_joint[i], rf[i]))
    #for i in range(len(convnet)):
    #    print("layer name = %s, output size = (%3d, %3d), RF size = %3d" % (layer_name[i], out_r[i], out_c[i], rf[i]))
