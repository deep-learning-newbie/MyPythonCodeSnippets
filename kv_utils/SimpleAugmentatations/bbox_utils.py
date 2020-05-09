import numpy as np
import random

def get_corners(bboxes):
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

def augment_rst_corners(RST_Mat, corners):
    corners = corners.reshape(-1, 2)
    corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))
    rst_bboxes = np.dot(RST_Mat, corners.T).T
    rst_bboxes = rst_bboxes.reshape(-1, 8)
    return rst_bboxes

def get_augmented_rois(rst_corners, img_w, img_h):
    xcordinates = rst_corners[:, [0, 2, 4, 6]]
    ycordinates = rst_corners[:, [1, 3, 5, 7]]
    ##cliping oof the cordinates if out of image
    xcordinates[xcordinates < 0] = 0
    xcordinates[xcordinates > img_w] = img_w - 1
    ycordinates[ycordinates < 0] = 0
    ycordinates[ycordinates > img_h] = img_h - 1

    xmin = np.min(xcordinates, 1).reshape(-1, 1)
    ymin = np.min(ycordinates, 1).reshape(-1, 1)
    xmax = np.max(xcordinates, 1).reshape(-1, 1)
    ymax = np.max(ycordinates, 1).reshape(-1, 1)
    augmented_rois = np.hstack((xmin, ymin, xmax, ymax))
    return augmented_rois
