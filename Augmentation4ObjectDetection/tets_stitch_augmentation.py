import os
import sys
import cv2
import numpy as np
import random
from Augmentation4ObjectDetection.GeometricalAugmentations.StitchAugmentation import StitchAugmentation
import pandas as pd
import FileUtils.img_roi_adapters as im_adpater

random.seed(18)

with open('img_folder/test_files.txt', 'r') as f:
    lines = [line.rstrip() for line in f]
List_Gts = lines
random.shuffle(List_Gts)

def draw_pascal_rois_lbl(img, rois, lbls):
    color_roi = (255, 0, 0)
    color_font = (0, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    line_thickness = 2
    num_instance = rois.shape[0]
    roi_lbls = lbls.flatten()
    for ii in range(num_instance):
        pt1 = (int(rois[ii][0]), int(rois[ii][1]))
        pt2 = (int(rois[ii][2]), int(rois[ii][3]))
        pt3 = (int(rois[ii][0]) + 5, int(rois[ii][1]) + 5)
        lbl_str = str(int(roi_lbls[ii]))
        img = cv2.rectangle(img, pt1, pt2, color_roi, int(2))
        img = cv2.putText(img, lbl_str, pt3, font,
                    fontScale, color_font, line_thickness, cv2.LINE_AA)

    return img
    #cv2.namedWindow("ImageRoi", cv2.WINDOW_NORMAL)
    #cv2.imshow("ImageRoi", img)
    #cv2.waitKey(0)

def parsefile2numpy(file_path):
    df_numpy = pd.read_csv(file_path, header=None, delim_whitespace=True).to_numpy()
    return df_numpy

def get_pascal_rois_lbls(gt_file, img_shape):
    data_np = parsefile2numpy(gt_file)
    lbls = data_np[:, 0]
    pascal_lbl = lbls.flatten()
    rois = data_np[:, 1:]
    pascal_roi = im_adpater.yolo2pascal_rois(rois, img_shape)
    return pascal_roi, pascal_lbl

def dump_gt_img_roi(img, roi, lbl, img_index, prefix = None):
    if prefix is None:
        prefix = ''
    img = draw_pascal_rois_lbl(img, roi, lbl)
    img_name = prefix + "%6.6d" % (img_index) + '.png'
    img_path = os.path.join('img_folder', img_name)
    cv2.imwrite(img_path, img)

def disp_stitch_roi():
    for ii in range (0, len(List_Gts)-4, 4):
        gt0, gt1, gt2, gt3 = List_Gts[ii], List_Gts[ii + 1], List_Gts[ii + 2], List_Gts[ii + 3]
        img_path0 = gt0.replace('txt', 'jpg')
        img_path1 = gt1.replace('txt', 'jpg')
        img_path2 = gt2.replace('txt', 'jpg')
        img_path3 = gt3.replace('txt', 'jpg')

        img0 = cv2.imread(img_path0)
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)
        img3 = cv2.imread(img_path3)
        list4img = [img0, img1, img2, img3]

        pascal_roi0, lbl0 = get_pascal_rois_lbls(gt0, img0.shape)
        pascal_roi1, lbl1 = get_pascal_rois_lbls(gt1, img1.shape)
        pascal_roi2, lbl2 = get_pascal_rois_lbls(gt2, img2.shape)
        pascal_roi3, lbl3 = get_pascal_rois_lbls(gt3, img3.shape)

        dump_gt_img_roi(img0, pascal_roi0, lbl0, ii)
        dump_gt_img_roi(img1, pascal_roi1, lbl1, ii + 1)
        dump_gt_img_roi(img2, pascal_roi2, lbl2, ii + 2)
        dump_gt_img_roi(img3, pascal_roi3, lbl3, ii + 3)
        obj_stitch = StitchAugmentation(img0.shape)
        list4rois = [pascal_roi0, pascal_roi1, pascal_roi2, pascal_roi3]
        list4lbls = [lbl0, lbl1, lbl2, lbl3]
        aug_img, aug_roi, aug_lbls = obj_stitch.process_augmentation(list4img, list4rois, list4lbls)
        dump_gt_img_roi(aug_img, aug_roi, aug_lbls, ii + 4, 'stitch')
        print('Break')

disp_stitch_roi()
