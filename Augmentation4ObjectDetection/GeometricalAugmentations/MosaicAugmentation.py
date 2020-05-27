import cv2
import numpy as np
import random

class MosaicAugmentation(object):
    def __init__(self, network_img_size):
        self.network_img_size = network_img_size

    def get_mosaic_center(self):
        net_sz = self.network_img_size
        xc, yc = [int(random.uniform(net_sz * 0.5, net_sz * 1.5))
                  for _ in range(2)]
        return xc, yc

    def get_resized_image(self, img):
        h0, w0 = img.shape[:2]  # orig hw
        ratio = self.network_img_size / max(h0, w0)  # resize image to network_size
        img_rsz = cv2.resize(img, (int(w0 * ratio), int(h0 * ratio)), cv2.INTER_LINEAR)
        return img_rsz, (h0, w0)  # img, hw_original

    def get_resized_translated_rois(self):
        pass

    def process_augmentation(self, list4imgs, list4rois=None):
        assert len(list4imgs) == 4, 'Need 4 images to build MOSAIC'
        #assert len(list4rois) != 4, 'Need 4 rois to build MOSAIC'

        net_sz = self.network_img_size
        xc, yc = self.get_mosaic_center()
        img4 = np.full((net_sz * 2, net_sz * 2, 3), 128, dtype=np.uint8)  # base image with 4 tiles
        for ii, index in enumerate(range(len(list4imgs))):
            img_original = list4imgs[ii]
            img_rsz, (orig_h, orig_w) = self.get_resized_image(img_original)
            rsz_h, rsz_w = img_rsz.shape[:2]

            # place img in img4
            if ii == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - rsz_w, 0), max(yc - rsz_h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = rsz_w - (x2a - x1a), rsz_h - (y2a - y1a), rsz_w, rsz_h  # xmin, ymin, xmax, ymax (small image)
            elif ii == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - rsz_h, 0), min(xc + rsz_w, net_sz * 2), yc
                x1b, y1b, x2b, y2b = 0, rsz_h - (y2a - y1a), min(rsz_w, x2a - x1a), rsz_h
            elif ii == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - rsz_w, 0), yc, xc, min(net_sz * 2, yc + rsz_h)
                x1b, y1b, x2b, y2b = rsz_w - (x2a - x1a), 0, max(xc, rsz_w), min(y2a - y1a, rsz_h)
            elif ii == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + rsz_w, net_sz * 2), min(net_sz * 2, yc + rsz_h)
                x1b, y1b, x2b, y2b = 0, 0, min(rsz_w, x2a - x1a), min(y2a - y1a, rsz_h)

            img4[y1a:y2a, x1a:x2a] = img_rsz[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

        return img4


