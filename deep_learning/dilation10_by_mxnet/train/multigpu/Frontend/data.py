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
                crop_size=628, rgb_mean=(117, 117, 117),
                data_name="data",
                label_name="softmax_label",
                margin=186, batch_size=1):
        super(FileIter, self).__init__()
        self.root_dir = root_dir
        self.flist_name = os.path.join(self.root_dir, flist_name)
        self.up = up
        self.batch_size = batch_size
        self.mean = np.array(rgb_mean).reshape(1, 1, 3)
        self.crop_size = crop_size
        self.margin = margin
        self.exact_crop = crop_size-2*margin
        assert self.exact_crop%8 == 0, "The crop size must be a multiple of 8 after removing the margin"
        self.data_name = data_name
        self.label_name = label_name

        self.flist = open(self.flist_name, 'r').readlines()
        self.num_data = len(self.flist)
        # ------------------------------------------------
        self.max_iter = self.num_data // self.batch_size
        self.iter = 0;

        self.shuffle()
        self.cursor = 0

        self.data = {}
        self.data = np.zeros((self.batch_size, 3, self.crop_size,self.crop_size))
        if up:
            self.label = np.zeros((self.batch_size, self.exact_crop*self.exact_crop))
        else:
            self.label = np.zeros((self.batch_size, self.exact_crop//8*self.exact_crop//8))

    def shuffle(self):
        tmp = np.random.permutation(self.num_data)
        new_list = []
        for i in xrange(self.num_data):
            new_list.append(self.flist[tmp[i]])

        self.flist = new_list


    def padImage(self, img, min_size, value):
        if len(img.shape)==3:
            h, w, _ = img.shape
        else:
            h, w = img.shape
        if h >= min_size and w >= min_size:
            return img
        top = bottom = left = right = 0
        if h < min_size:
            top = (min_size-h) / 2
            bottom = min_size - h -top

        if w < min_size:
            left = (min_size-w) / 2
            right = min_size - w -left

        if value < 0:
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT_101)
        else:
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value)
        return img

    def extendLabelMargin(self, img, margin_w, margin_h, value):
        if value < 0:
            img = cv2.copyMakeBorder(img, margin_h, margin_h, margin_w, margin_w, cv2.BORDER_REFLECT_101)
        else:
            img = cv2.copyMakeBorder(img, margin_h, margin_h, margin_w, margin_w, cv2.BORDER_CONSTANT, value=value)
        return img


    def _read(self):
        for i in xrange(self.batch_size):
            _, data_img_name, label_img_name = self.flist[self.cursor].strip('\n').split('\t')

            self.data[i][:], self.label[i][:]= self._read_img(data_img_name, label_img_name)
            self.cursor += 1
            #if self.cursor == self.num_data:
            #    self.reset()

    def _read_img(self, img_name, label_name):
        img = cv2.imread(img_name, 1).astype(np.float32)
        label = cv2.imread(label_name, 0).astype(np.float32)
        assert img.shape[0:2] == label.shape, "the image size and label size don't match"

        img = self.extendLabelMargin(img, self.margin, self.margin, -1)
        img = self.padImage(img, self.crop_size, -1)
        label = self.extendLabelMargin(label, self.margin, self.margin, 255)
        label = self.padImage(label, self.crop_size, 255)
        h, w, _ = img.shape
        rand_w = np.random.randint(0, w-self.crop_size+1)
        rand_h = np.random.randint(0, h-self.crop_size+1)
        crop_img = img[rand_h:rand_h+self.crop_size, rand_w:rand_w+self.crop_size, :]
        crop_label = label[rand_h:rand_h+self.crop_size, rand_w:rand_w+self.crop_size]
        crop_img -= self.mean
        crop_img = np.transpose(crop_img, (2, 0, 1))
        if self.up:
            crop_label = crop_label[self.margin:self.margin+self.exact_crop, self.margin:self.margin+self.exact_crop]
        else:
            crop_label = crop_label[self.margin:self.margin+self.exact_crop:8, self.margin:self.margin+self.exact_crop:8]


        return (crop_img, crop_label.ravel())


    @property
    def provide_data(self):
        """ the name and shape of data provided by this iterator """
        # [1] here is batch, here just 1. here we just have one data, len()=1
        return [(self.data_name, (self.batch_size, 3, self.crop_size, self.crop_size))]

    @property
    def provide_label(self):
        if self.up:
            return [(self.label_name, (self.batch_size, self.exact_crop*self.exact_crop))]
        else:
            return [(self.label_name, (self.batch_size, self.exact_crop//8*self.exact_crop//8))]


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
