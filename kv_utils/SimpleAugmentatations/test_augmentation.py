import cv2
import numpy as np
from SimpleAugmentatations.GeometricalTransformations import Geometrical_Transformation
from SimpleAugmentatations.NoiseTransformations import Noise_Transformations

img_path = 'img_folder/iron_man.png'
img_path1 = 'img_folder/iron_man_roi.png'
img_aug_path = 'img_folder/iron_man_aug.png'



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


#test_flip()
#test_rotate()
#test_rotate_roi()
test_noise()


