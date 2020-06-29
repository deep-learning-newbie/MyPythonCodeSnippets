import numpy as np
import cv2
from PIL import Image

def pil2opencv_image(pil_img):
    numpy_image = np.array(pil_img)
    # convert to a openCV2 image, notice the COLOR_RGB2BGR which means that
    # the color is converted from RGB to BGR format
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return opencv_image

def opencv2pil_image(opencv_image):
    # convert from openCV2 to PIL. Notice the COLOR_BGR2RGB which means that
    # the color is converted from BGR to RGB
    pil_image = Image.fromarray(
        cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    )
    return pil_image

def pascal2yolo_rois(pascal_rois, image_shape):
    img_ht, img_wd = image_shape[:2]
    x1 = pascal_rois[:, 0]; y1 = pascal_rois[:, 1]
    x2 = pascal_rois[:, 2]; y2 = pascal_rois[:, 3]

    xmid = (x1 + x2) / 2; ymid = (y1 + y2) / 2
    wd = (x2 - x1) / 2; ht = (y2 -y1) / 2

    xmid = xmid / img_wd; ymid = ymid / img_ht
    wd = wd / img_wd; ht = ht / img_ht

    yolo_roi = np.hstack([np.expand_dims(xmid, axis=1), np.expand_dims(ymid, axis=1),
                      np.expand_dims(wd, axis=1), np.expand_dims(ht, axis=1)])

    return yolo_roi

def yolo2pascal_rois(yolo_rois, image_shape):
    img_ht, img_wd = image_shape[:2]

    xmid = yolo_rois[:, 0]
    ymid = yolo_rois[:, 1]
    wd_half = yolo_rois[:, 2] / 2
    ht_half = yolo_rois[:, 3] / 2

    x1 = (xmid - wd_half) * img_wd
    y1 = (ymid - ht_half) * img_ht
    x2 = (xmid + wd_half) * img_wd
    y2 = (ymid + ht_half) * img_ht

    pascal_rois = np.hstack([np.expand_dims(x1, axis=1), np.expand_dims(y1, axis=1),
                      np.expand_dims(x2, axis=1), np.expand_dims(y2, axis=1)])

    return pascal_rois
