import cv2
import numpy as np
import random

class StitchAugmentation(object):
    def __init__(self, standard_img_shape):
        self.std_img_ht = standard_img_shape[0] if standard_img_shape[0] % 2 == 0 else standard_img_shape[0] + 1
        self.std_img_wd = standard_img_shape[1] if standard_img_shape[1] % 2 == 0 else standard_img_shape[1] + 1
        self.chnls = standard_img_shape[2]
        self.stitch_size_dims = (self.std_img_ht, self.std_img_wd, self.chnls)
    def get_resized_translated_rois(self, index, original_shape, orig_bboxes):
        wd_ratio = (self.std_img_wd / 2) / original_shape[1]
        ht_ratio = (self.std_img_ht / 2) / original_shape[0]
        rsz_bboxes = orig_bboxes.copy().astype(np.float32)

        ##resize rois
        rsz_bboxes[:, 0] *= wd_ratio  # width x1
        rsz_bboxes[:, 2] *= wd_ratio  # width x2
        rsz_bboxes[:, 1] *= ht_ratio  # ht_ratio y1
        rsz_bboxes[:, 3] *= ht_ratio  # ht_ratio y2
        ##translate factor
        if index == 0:  # top left
            pass  # no translation required
        if index == 1:  # top right
            rsz_bboxes[:, 0] = rsz_bboxes[:, 0] + self.std_img_wd / 2
            rsz_bboxes[:, 2] = rsz_bboxes[:, 2] + self.std_img_wd / 2
        if index == 2:  # bottom left
            rsz_bboxes[:, 1] = rsz_bboxes[:, 1] + self.std_img_ht / 2
            rsz_bboxes[:, 3] = rsz_bboxes[:, 3] + self.std_img_ht / 2
        if index == 3:  # bottom right
            rsz_bboxes[:, 0] = rsz_bboxes[:, 0] + self.std_img_wd / 2
            rsz_bboxes[:, 2] = rsz_bboxes[:, 2] + self.std_img_wd / 2
            rsz_bboxes[:, 1] = rsz_bboxes[:, 1] + self.std_img_ht / 2
            rsz_bboxes[:, 3] = rsz_bboxes[:, 3] + self.std_img_ht / 2

        return rsz_bboxes

    def process_augmentation(self, list4imgs, list4rois=None):
        assert len(list4imgs) == 4, 'Need 4 images to build MOSAIC'
        assert len(list4rois) == 4, 'Need 4 rois to build MOSAIC'
        aug_img = np.ones(self.stitch_size_dims, dtype="uint8")
        aug_rois = np.zeros((1, 4), dtype=float)
        for ii, index in enumerate(range(len(list4imgs))):
            orig_img = list4imgs[ii]
            orig_roi = list4rois[ii]
            rsz_img = cv2.resize(orig_img, (self.std_img_wd // 2, self.std_img_ht// 2))
            rsz_ht, rsz_wd = rsz_img.shape[:2]

            if orig_roi is not None:
                rsz_rois = self.get_resized_translated_rois(ii, orig_img.shape, orig_roi)
                aug_rois = np.vstack((rsz_rois, aug_rois))

            if ii == 0: ##top left
                bx1 = 0; by1 = 0; bx2 = rsz_wd; by2 = rsz_ht
            if ii == 1:  ##top right
                bx1 = rsz_wd; by1 = 0; bx2 = self.std_img_wd; by2 = rsz_ht
            if ii == 2: ##bottom left
                bx1 = 0; by1 = rsz_ht; bx2 = rsz_wd; by2 = self.std_img_ht
            if ii == 3:  ##bottom right
                bx1 = rsz_wd; by1 = rsz_ht; bx2 = self.std_img_wd; by2 = self.std_img_ht

            aug_img[by1:by2, bx1:bx2] = rsz_img
        aug_rois = aug_rois[:-1, :]
        return aug_img, aug_rois


