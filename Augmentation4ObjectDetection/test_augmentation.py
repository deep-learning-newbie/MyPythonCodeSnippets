import os
import sys
import cv2
import numpy as np
from GeometricalAugmentations.RotateScaleTranslate import RandomRST_Transformation
from GeometricalAugmentations.RandomRescale import RandomReScale
from GeometricalAugmentations.RandomFlipShear import RandomHorizontalFlip
from GeometricalAugmentations.RandomFlipShear import RandomShear

lib_path = os.path.join(os.path.realpath("."), "data_aug")
#sys.path.append(lib_path)

img_path = 'img_folder/iron_man.png'
img_path1 = 'img_folder/iron_man_roi.png'
img_aug_path = 'img_folder/iron_man_aug.png'

def draw_rect(im, cords, color=None):
    im = im.copy()

    cords = cords[:, :4]
    cords = cords.reshape(-1, 4)
    if not color:
        color = [255, 255, 255]
    for cord in cords:
        pt1, pt2 = (cord[0], cord[1]), (cord[2], cord[3])

        pt1 = int(pt1[0]), int(pt1[1])
        pt2 = int(pt2[0]), int(pt2[1])

        im = cv2.rectangle(im.copy(), pt1, pt2, color, int(2))
    return im

def test_shear():
    rois = np.array([[397, 108, 465, 190],
                     [591, 128, 608, 164]])

    color = (255, 0, 0)
    img = cv2.imread(img_path)
    iron_man_roi = draw_rect(img, rois, color)
    cv2.imwrite(img_path1, iron_man_roi)

    obj_shear = RandomShear(0.2, 1.0)
    aug_img, aug_rois = obj_shear.process_augmentation(img, rois)
    color_red = (0, 0, 255)
    aug_img = draw_rect(aug_img, aug_rois, color_red)
    cv2.imwrite(img_aug_path, aug_img)


def test_horizontal_flip():
    rois = np.array([[397, 108, 465, 190],
                     [591, 128, 608, 164]])

    color = (255, 0, 0)
    img = cv2.imread(img_path)
    iron_man_roi = draw_rect(img, rois, color)
    cv2.imwrite(img_path1, iron_man_roi)

    obj_hflip = RandomHorizontalFlip(1.0)
    aug_img, aug_rois = obj_hflip.process_augmentation(img, rois)
    color_red = (0, 0, 255)
    aug_img = draw_rect(aug_img, aug_rois, color_red)
    cv2.imwrite(img_aug_path, aug_img)

def test_rescale():
    rois = np.array([[397, 108, 465, 190],
                     [591, 128, 608, 164]])
    color = (255, 0, 0)

    img = cv2.imread(img_path)
    iron_man_roi = draw_rect(img, rois, color)
    cv2.imwrite(img_path1, iron_man_roi)
    obj_rsz = RandomReScale(0.1)
    aug_img, aug_rois = obj_rsz.process_augment(img, rois)
    color_red = (0, 0, 255)
    aug_img = draw_rect(aug_img, aug_rois, color_red)
    cv2.imwrite(img_aug_path, aug_img)

def test_rst():
    rois = np.array([[397, 108, 465, 190],
                     [591, 128, 608, 164]])
    color = (255, 0, 0)

    img = cv2.imread(img_path)
    iron_man_roi = draw_rect(img, rois, color)
    cv2.imwrite(img_path1, iron_man_roi)

    obj_rst = RandomRST_Transformation(15, 0.1, 0.1, 0.1)
    aug_img, aug_rois = obj_rst.random_rotate_scale_translate(img, rois)
    color_red = (0, 0, 255)
    aug_img = draw_rect(aug_img, aug_rois, color_red)
    cv2.imwrite(img_aug_path, aug_img)

#test_rst()
#test_rescale()
#test_horizontal_flip()
test_shear()


