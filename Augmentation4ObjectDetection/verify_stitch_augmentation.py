import os
import sys
import cv2
import numpy as np
from Augmentation4ObjectDetection.GeometricalAugmentations.StitchAugmentation import StitchAugmentation


with open('img_folder/test_files.txt', 'r') as f:
    lines = [line.rstrip() for line in f]
List_images = lines

def parse_gt_file(gt_file_path):
    with open(gt_file_path, 'r') as f:
        lines = f.readlines()
    if not lines:
        return None

    lines = [line.split(' ') for line in lines if len(line) > 0]

    out_data = []
    for line in lines:
        line_entries = [float(entry) for entry in line]
        line_entries = [line_entries[1], line_entries[2], line_entries[3], line_entries[4], line_entries[0]]
        out_data.append(line_entries)
    return out_data

def get_opencv_compatible_roi(yolo_roi, img_shape):
    img_ht, img_wd, chnls = img_shape
    opencv_rois = []
    for roi in yolo_roi:
        xmid, ymid, wd, ht, lbl = roi[0] * img_wd, roi[1] * img_ht, roi[2] * img_wd, \
                                  roi[3] * img_ht, roi[4]
        x1 = int(xmid - wd/2)
        y1 = int(ymid - ht/2)
        x2 = int(xmid + wd/2)
        y2 = int(ymid + ht/2)
        opencv_rois.append([(x1, y1), (x2, y2), lbl])

    return opencv_rois

def draw_roi(img, gt_data):
    color_roi = (255, 0, 0)
    color_font = (0, 255, 255)
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale
    fontScale = 1
    # Line thickness of 2 px
    thickness = 2
    for roi in gt_data:
        img = cv2.rectangle(img, roi[0], roi[1], color_roi, int(2))
        lbl_str = str(roi[2])
        x = roi[0][0] + 5
        y = roi[0][1] + 5
        cv2.putText(img, lbl_str, (x, y), font,
                    fontScale, color_font, thickness, cv2.LINE_AA)
    return img

def disp_image_roi():
    for index, img_path in enumerate(List_images):
        img = cv2.imread(img_path)
        gt_name = img_path.replace('jpg', 'txt')
        gt = get_opencv_compatible_roi(parse_gt_file(gt_name), img.shape)
        img = draw_roi(img, gt)
        new_img_name = "%6.6d.jpg" % (index)
        new_img_path = os.path.join('img_folder', new_img_name)
        cv2.imwrite(new_img_path, img)

    print('Exit-disp_image_roi')

def get_stitch_compatible_rois(list_rois_lbl, img_shape):
    img_ht, img_wd, chnls = img_shape
    lst_roi = []
    lst_lbl = []

    for roi in list_rois_lbl:
        xmid, ymid, wd, ht, lbl = roi[0] * img_wd, roi[1] * img_ht, roi[2] * img_wd, \
                                  roi[3] * img_ht, roi[4]
        x1 = int(xmid - wd / 2)
        y1 = int(ymid - ht / 2)
        x2 = int(xmid + wd / 2)
        y2 = int(ymid + ht / 2)
        lst_roi.append([x1, y1, x2, y2])
        lst_lbl.append(lbl)

    stitch_roi = np.array(lst_roi)
    stitch_lbl = np.array(lst_lbl).reshape(-1, 1)

    return stitch_roi, stitch_lbl

def draw_rect_lbl(im, cords, lbls, color=None):
    im = im.copy()
    cords = cords[:, :4]
    cords = cords.reshape(-1, 4)

    color_font = (0, 255, 255)
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale
    fontScale = 1
    # Line thickness of 2 px
    thickness = 2

    if not color:
        color = [255, 255, 255]
    index = 0
    for cord in cords:
        pt1, pt2 = (cord[0], cord[1]), (cord[2], cord[3])
        pt1 = int(pt1[0]), int(pt1[1])
        pt2 = int(pt2[0]), int(pt2[1])

        lbl_str = str(lbls[index][0])
        x = int(cord[0] + 5)
        y = int(cord[1] + 5)

        im = cv2.rectangle(im.copy(), pt1, pt2, color, int(2))
        cv2.putText(im, lbl_str, (x, y), font,
                    fontScale, color_font, thickness, cv2.LINE_AA)
        index += 1
    return im

disp_image_roi()

def disp_stitch_roi():
    list4img = []
    list4rois = []
    list4lbls = []
    for index, img_path in enumerate(List_images):
        img = cv2.imread(img_path)
        gt_name = img_path.replace('jpg', 'txt')

        list4img.append(img)
        stitch_roi, stitch_lbl = get_stitch_compatible_rois(parse_gt_file(gt_name), img.shape)
        list4rois.append(stitch_roi)
        list4lbls.append(stitch_lbl)

    obj_stitch = StitchAugmentation((1080, 1920, 3))
    aug_img, aug_roi, aug_lbls = obj_stitch.process_augmentation(list4img, list4rois, list4lbls)
    im = draw_rect_lbl(aug_img, aug_roi, aug_lbls)
    cv2.imwrite('img_folder/stitch_img.jpg', im)

disp_stitch_roi()
