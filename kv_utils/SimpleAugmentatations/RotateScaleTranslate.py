import cv2
import numpy as np
import random
import math
from bbox_utils import *
#random.seed(18)

class RandomRST_Transformation(object):
    def __init__(self, angle_range, scale_range, tx, ty, p=0.5):
        self.p = p
        self.angle_range = abs(angle_range)
        self.scale_range = scale_range
        self.tx = tx
        self.ty = ty
        self.scale_range = scale_range

    def random_rotate_scale_translate(self, image, bboxes=None):
        if True:
        #if random.random() > self.p:
            if self.scale_range > 1 or self.tx > 1 or self.ty > 1:
                print('Input Value out of range')
                exit(-1)
            (h, w) = image.shape[:2]
            (cx, cy) = (w // 2, h // 2)
            angle = int(random.uniform(-self.angle_range, self.angle_range))
            scale = random.uniform(1 - self.scale_range, 1 + self.scale_range)
            RST = cv2.getRotationMatrix2D((cx, cy), angle, scale)
            tx = int(random.uniform(0, self.tx) * w)
            ty = int(random.uniform(0, self.ty) * h)
            RST[0, 2] += - tx
            RST[1, 2] += - ty
            affine_image = cv2.warpAffine(image, RST, (w, h))
            if bboxes is not None:
                corners = get_corners(bboxes)
                rst_corners = augment_rst_corners(RST, corners)
                rst_rois = get_augmented_rois(rst_corners, w, h)
            return affine_image, rst_rois
        else:
            return image.copy(), bboxes.copy()


    def get_corners(self, bboxes):
        xmin = bboxes[:, 0].reshape(-1, 1)
        ymin = bboxes[:, 1].reshape(-1, 1)
        xmax = bboxes[:, 2].reshape(-1, 1)
        ymax = bboxes[:, 3].reshape(-1, 1)
        x1 = xmin
        y1 = ymin
        x2 = xmax
        y2 = ymin
        x3 = xmax
        y3 = ymax
        x4 = xmin
        y4 = ymax
        corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))
        return corners

    def augment_rst_corners(self, RST_Mat, corners):
        corners = corners.reshape(-1, 2)
        corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))
        rst_bboxes = np.dot(RST_Mat, corners.T).T
        rst_bboxes = rst_bboxes.reshape(-1, 8)
        return rst_bboxes

    def get_augmented_rois(self, rst_corners, img_w, img_h):
        xcordinates = rst_corners[:, [0, 2, 4, 6]]
        ycordinates = rst_corners[:, [1, 3, 5, 7]]
        ##cliping of the cordinates if out of image
        xcordinates[xcordinates < 0] = 0
        xcordinates[xcordinates > img_w] = img_w-1
        ycordinates[ycordinates < 0] = 0
        ycordinates[ycordinates > img_h] = img_h-1

        xmin = np.min(xcordinates, 1).reshape(-1, 1)
        ymin = np.min(ycordinates, 1).reshape(-1, 1)
        xmax = np.max(xcordinates, 1).reshape(-1, 1)
        ymax = np.max(ycordinates, 1).reshape(-1, 1)
        augmented_rois = np.hstack((xmin, ymin, xmax, ymax))
        return augmented_rois

class Resize(object)
    def __init__(self):

##test


