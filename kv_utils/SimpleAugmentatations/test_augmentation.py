import os
import sys
import cv2
import numpy as np
from GeometricalTransformations import Geometrical_Transformation
from NoiseTransformations import Noise_Transformations
from RotateScaleTranslate import RandomRST_Transformation

lib_path = os.path.join(os.path.realpath("."), "data_aug")
#sys.path.append(lib_path)

img_path = 'img_folder/iron_man.png'
img_path1 = 'img_folder/iron_man_roi.png'
img_aug_path = 'img_folder/iron_man_aug.png'


def draw_rect(im, cords, color = None):
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

def test_flip():
    img = cv2.imread(img_path)
    obj_geo = Geometrical_Transformation(img)
    flip_img = obj_geo.flip_image('hv')
    cv2.imwrite(img_aug_path, flip_img)

def test_rotate():
    img = cv2.imread(img_path)
    obj_geo = Geometrical_Transformation(img)
    rot_img = obj_geo.rotate_image_bound(30)
    cv2.imwrite(img_aug_path, rot_img)

def test_rotate_roi():
    start_point = (92, 285)
    end_point = (117, 311)
    color = (255, 0, 0)
    thickness = 2
    img = cv2.imread(img_path)
    obj_geo = Geometrical_Transformation(img)
    rot_img = obj_geo.rotate_image_bound(30)
    corners = np.hstack((92, 285, 117, 285, 117, 311, 92, 311))
    rot_corners = obj_geo.rotate_roi(corners, 30)
    start_point = tuple(np.amin(rot_corners, axis=0))
    end_point = tuple(np.amax(rot_corners, axis=0))
    rot_img = cv2.rectangle(rot_img, start_point, end_point, color, thickness)
    cv2.imwrite(img_aug_path, rot_img)


def test_noise():
    img = cv2.imread(img_path)
    obj_noise = Noise_Transformations(img)
    noise_img = obj_noise.noisy_img('sp')
    cv2.imwrite(img_aug_path, noise_img)
    print('ss')

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

test_rst()
#test_flip()
#test_rotate()
#test_rotate_roi()
#test_noise()


