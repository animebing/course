#!/usr/bin/python

from __future__ import print_function, division
import numpy as np
import os
import argparse
import cv2

classes=["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",\
       "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
category=["flat", "construction", "object", "nature", "sky", "human", "vehicle"]
category2class={"flat":[0, 1], "construction":[2, 3, 4], "object":[5, 6, 7], "nature":[8, 9], "sky":[10], "human":[11, 12], "vehicle":[13, 14, 15, 16, 17, 18]}

def computeClassIOU(gt_list, output_list):
    inter = np.zeros((len(classes),))
    union = np.zeros((len(classes),))
    for i in xrange(len(gt_list)):
        gt_img = cv2.imread(gt_list[i].strip('\n'), 0)
        output_img = cv2.imread(output_list[i].strip('\n'), 0)
        ignore_idx = np.where(gt_img == 255)
        output_img[ignore_idx] = 255
        for j in xrange(len(classes)):
            gt_logic = (gt_img == j)
            output_logic = (output_img == j)
            inter[j] += np.logical_and(gt_logic, output_logic).sum()
            union[j] += np.logical_or(gt_logic, output_logic).sum()

    return inter/union

def computeCateIOU(gt_list, output_list):
    inter = np.zeros((len(category),))
    union = np.zeros((len(category),))
    for i in xrange(len(gt_list)):
        gt_img = cv2.imread(gt_list[i].strip('\n'), 0)
        output_img = cv2.imread(output_list[i].strip('\n'), 0)
        ignore_idx = np.where(gt_img == 255)
        output_img[ignore_idx] = 255
        for j in xrange(len(category)):
            gt_logic = np.zeros(gt_img.shape, dtype=bool)
            output_logic = np.zeros(gt_img.shape, dtype=bool)
            for t in category2class[category[j]]:
                gt_tmp = np.where(gt_img==t)
                gt_logic[gt_tmp] = True
                output_tmp = np.where(output_img==t)
                output_logic[output_tmp] = True

            inter[j] += np.logical_and(gt_logic, output_logic).sum()
            union[j] += np.logical_or(gt_logic, output_logic).sum()

    return inter/union

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True,
                        help="directory to store ground truth image and segmented image")
    parser.add_argument('--gt_list', type=str, required=True,
                        help="ground truth image list")
    parser.add_argument('--output_list', type=str, required=True,
                        help="segmented image list")

    args = parser.parse_args()
    gt_list = open(os.path.join(args.img_dir, args.gt_list), 'r').readlines()
    output_list = open(os.path.join(args.img_dir, args.output_list), 'r').readlines()

    class_iou = computeClassIOU(gt_list, output_list)
    mean_class = class_iou.sum()/len(class_iou)

    cate_iou = computeCateIOU(gt_list, output_list)
    mean_cate = cate_iou.sum()/len(cate_iou)

    classDict = dict(zip(classes, class_iou))
    classDict["mean"] = mean_class
    cateDict = dict(zip(category, cate_iou))
    cateDict["mean"] = mean_cate
    print(classDict)
    print(cateDict)
