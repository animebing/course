#!/usr/bin/python
from __future__ import print_function
import mxnet as mx
import numpy as np

class ClassAccuracy(mx.metric.EvalMetric):
    """
    My own accuracy metric for segmentation
    """
    def __init__(self):
        super(ClassAccuracy, self).__init__('ClassAccuracy')

    def update(self, label, pred):
        """
        Update classification accuracy

        Args
        ----
        label: mx ndarray  batch_size * (32*32)
        pred:  mx ndarray batch_size * class * (32*32)
        """
        label = label.asnumpy().astype('int32')
        non_ignore_idxs = np.where(label != 255)


        pred_label = mx.nd.argmax_channel(pred).asnumpy().astype('int32')


        label = label[non_ignore_idxs]
        pred_label = pred_label[non_ignore_idxs]

        self.sum_metric += (pred_label==label).sum()
        self.num_inst += len(label)


class ClassCrossEntropy(mx.metric.EvalMetric):
    """
    my own cross entropy loss for segmentation
    """
    def __init__(self):
        super(ClassCrossEntropy, self).__init__('ClassCrossEntropy')

    def update(self, label, pred):

        label = label.asnumpy().astype('int32')
        label = label.ravel()

        pred = np.transpose(pred.asnumpy(), (0, 2, 1))
        pred = np.reshape(pred, (-1, pred.shape[2]))

        # non_ignore_idxs is a tuple of numpy array
        non_ignore_idxs = np.where(label!=255)
        label = label[non_ignore_idxs[0]]
        pred = pred[non_ignore_idxs[0], :]
        tmp = pred[np.arange(len(label)), label]
        flag = np.where(tmp==0)
        if flag[0].size > 0:
            print("softmax 0")
            print(tmp[flag[0]])
        #print(pred.sum(axis=1))
        self.sum_metric += -np.log(pred[np.arange(len(label)), label]).sum()
        self.num_inst += len(label)







