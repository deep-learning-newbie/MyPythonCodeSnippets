import cv2
import numpy as np
import random
from .RandomRescale import RandomReScale

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    def process_augmentation(self, image, bboxes=None):
        if random.random() < self.p:
            img_ht, img_wd = image.shape[:2]
            aug_img = cv2.flip(image, 1)
            flip_bboxes = None
            if bboxes is not None:
                flip_bboxes = bboxes.copy()
                flip_bboxes[:, 0] = img_wd - bboxes[:, 0] - 1 ## pixel index start from 0 hence -1
                flip_bboxes[:, 2] = img_wd - bboxes[:, 2] - 1 ## pixel index start from 0 hence -1
                x1 = flip_bboxes[:, 2].copy()
                x2 = flip_bboxes[:, 0].copy()
                flip_bboxes[:, 0] = x1
                flip_bboxes[:, 2] = x2
            return aug_img, flip_bboxes
        else:
            return image, bboxes

class RandomShear(object):
    def __init__(self, shear_range, p=0.5):
        self.shear_range = (-shear_range, shear_range)
        self.p = p
    def process_augmentation(self, image, bboxes=None):
        if random.random() < self.p:
            shear_factor = random.uniform(*self.shear_range)
            img_ht, img_wd = image.shape[:2]
            aug_img = None; aug_bboxes = None
            if shear_factor < 0:
                aug_img, aug_bboxes = RandomHorizontalFlip(1.0).process_augmentation(image, bboxes)
            if bboxes is not None:
                aug_bboxes[:, [0, 2]] += ((aug_bboxes[:, [1, 3]]) * abs(shear_factor)).astype(int)

            M = np.array([[1, abs(shear_factor), 0], [0, 1, 0]])
            nW = image.shape[1] + abs(shear_factor * image.shape[0])

            aug_img = cv2.warpAffine(aug_img, M, (int(nW), aug_img.shape[0]))
            if shear_factor < 0:
                aug_img, aug_bboxes = RandomHorizontalFlip(1.0).process_augmentation(aug_img, aug_bboxes)

            aug_img = cv2.resize(aug_img, (img_wd, img_ht))
            scale_factor_x = nW / img_wd
            aug_bboxes = aug_bboxes.astype(np.float32)
            aug_bboxes[:, 0] *= scale_factor_x
            aug_bboxes[:, 2] *= scale_factor_x
            return aug_img, aug_bboxes
        else:
            return image, bboxes







