import cv2
import numpy as np
import random
import math
from GeometricalAugmentations.bbox_utils import *

class RandomReScale(object):
    def __init__(self, scale_range, p=0.5):
        self.scale_range = scale_range
        self.p = p
    def process_augment(self, img, bboxes=None):
        if random.random() < self.p:
            img_ht, img_wd = img.shape[:2]
            scale = random.uniform(1 - self.scale_range, 1 + self.scale_range)
            new_img_wd, new_img_ht = (img_wd*scale, img_ht*scale)
            rsz_img = cv2.resize(img, (int(new_img_wd), int(new_img_ht)))
            rsz_bboxes = None
            if bboxes is not None:
                rsz_bboxes = bboxes.copy().astype(np.float32)
                rsz_bboxes[:, 0] *= (new_img_wd / img_wd)  # width x1
                rsz_bboxes[:, 2] *= (new_img_wd / img_wd)  # width x2
                rsz_bboxes[:, 1] *= (new_img_ht / img_ht)  # height y1
                rsz_bboxes[:, 3] *= (new_img_ht / img_ht)  # height y2
            return rsz_img, rsz_bboxes
        else:
            return img, bboxes










