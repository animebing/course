# pylint: skip-file
""" file iterator for dilation convolution """

import mxnet as mx
import numpy as np
import sys, os
from mxnet.io import DataIter
#from PIL import Image
import cv2

class FileIter(DataIter):
    """ FileIter object in dilation convolution. Taking a list file to get dataiter.
    Parameters
    ------------
    root_dir: string
        the root dir of image/label lie in
    flist_name: string
        the list file of iamge and label, every line owns the form:
        index \t image_data_path \t image_label_path
    crop_size: int
        randomly crop the input image with crop_size
    data_name: string
        the data name used in symbol, "data"(default data name)
    label_name: string
        the label name used in symbol, "softmax_label"(default label name)
    """
    def __init__(self, root_dir, flist_name, up,
                img_size=[1024, 2048], rgb_mean=(117, 117, 117),
                data_name="data",
                label_name="softmax_label",
                margin=186, batch_size=2):
        super(FileIter, self).__init__()
        self.root_dir = root_dir
        self.flist_name = os.path.join(self.root_dir, flist_name)
        self.up = up
        self.img_size = img_size

        self.batch_size = batch_size
        self.mean = np.array(rgb_mean).reshape(1, 1, 3)
        self.margin = margin

        self.data_name = data_name
        self.label_name = label_name

        self.flist = open(self.flist_name, 'r').readlines()
        self.num_data = len(self.flist)

        self.exact_batch = self.batch_size//2
        self.max_iter = self.num_data // self.exact_batch
        self.iter = 0;

        self.shuffle()
        self.cursor = 0

        self.data = np.zeros((self.batch_size, 3, self.img_size[0]+2*self.margin,self.img_size[1]//2+2*self.margin))
        if up:
            self.label = np.zeros((self.batch_size, self.img_size[0]*self.img_size[1]//2))
        else:
            self.label = np.zeros((self.batch_size, self.img_size[0]//8*self.img_size[1]//16))

    def shuffle(self):
        tmp = np.random.permutation(self.num_data)
        new_list = []
        for i in xrange(self.num_data):
            new_list.append(self.flist[tmp[i]])

        self.flist = new_list


    def extendLabelMargin(self, img, margin_w, margin_h, value):
        if value < 0:
            img = cv2.copyMakeBorder(img, margin_h, margin_h, margin_w, margin_w, cv2.BORDER_REFLECT_101)
        else:
            img = cv2.copyMakeBorder(img, margin_h, margin_h, margin_w, margin_w, cv2.BORDER_CONSTANT, value=value)
        return img


    def _read(self):
        for i in xrange(self.exact_batch):
            _, data_img_name, label_img_name = self.flist[self.cursor].strip('\n').split('\t')

            self.data[2*i:2*(i+1)][:], self.label[2*i:2*(i+1)][:]= self._read_img(data_img_name, label_img_name)
            self.cursor += 1

    def _read_img(self, img_name, label_name):
        img = cv2.imread(img_name, 1).astype(np.float32)
        label = cv2.imread(label_name, 0).astype(np.float32)
        assert img.shape[0:2] == label.shape, "the image size and label size don't match"

        img_1 = self.extendLabelMargin(img[:, :self.img_size[1]//2, :], self.margin, self.margin, -1)
        img_2 = self.extendLabelMargin(img[:, self.img_size[1]//2:, :], self.margin, self.margin, -1)

        label_1 = label[:, :self.img_size[1]//2]
        label_2 = label[:, self.img_size[1]//2:]

        img_1 -= self.mean
        img_1 = np.transpose(img_1, (2, 0, 1))
        img_1 = img_1[np.newaxis, :, :, :]
        img_2 -= self.mean
        img_2 = np.transpose(img_2, (2, 0, 1))
        img_2 = img_2[np.newaxis, :, :, :]
        img = np.concatenate([img_1, img_2], axis=0)

        if not self.up:
            label_1 = label_1[::8, ::8]
            label_2 = label_2[::8, ::8]
        label_1 = label_1.ravel()
        label_1 = label_1[np.newaxis, :]
        label_2 = label_2.ravel()
        label_2 = label_2[np.newaxis, :]
        label = np.concatenate([label_1, label_2], axis=0)


        return (img, label)


    @property
    def provide_data(self):
        """ the name and shape of data provided by this iterator """
        return [(self.data_name, self.data.shape)]

    @property
    def provide_label(self):
        return [(self.label_name, self.label.shape)]


    def get_batch_size(self):
        return self.batch_size

    def reset(self):
        self.iter = 0
        self.cursor = 0
        self.shuffle()

    def next(self):
        """ return one dict which contains "data" and "label" """
        self.iter += 1
        if self.iter <= self.max_iter:
            self._read()
            return mx.io.DataBatch(data=[mx.nd.array(self.data)], label=[mx.nd.array(self.label)])
        else:
            self.reset()
            raise StopIteration
