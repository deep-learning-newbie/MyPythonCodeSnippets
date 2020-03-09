import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import tempfile
import shutil
from PIL import Image
from Augmentor import Operations


def rotate_images(tmpdir, rot):
    original_dimensions = (800, 800)

    im_tmp = tmpdir.mkdir("subfolder").join('test.JPEG')
    im = Image.new('RGB', original_dimensions)
    im.save(str(im_tmp), 'JPEG')

    r = Operations.Rotate(probability=1, rotation=rot)
    im = [im]
    im_r = r.perform_operation(im)

    assert im_r is not None
    assert im_r[0].size == original_dimensions


def test_rotate_images_90(tmpdir):
    rotate_images(tmpdir, 90)


def test_rotate_images_180(tmpdir):
    rotate_images(tmpdir, 180)


def test_rotate_images_270(tmpdir):
    rotate_images(tmpdir, 270)


def test_rotate_images_custom_temp_files():

    original_dimensions = (800, 800)

    tmpdir = tempfile.mkdtemp()
    tmp = tempfile.NamedTemporaryFile(dir=tmpdir, suffix='.JPEG')
    im = Image.new('RGB', original_dimensions)
    im.save(tmp.name, 'JPEG')

    r = Operations.Rotate(probability=1, rotation=90)
    im = [im]
    im_r = r.perform_operation(im)

    assert im_r is not None
    assert im_r[0].size == original_dimensions

    tmp.close()
    shutil.rmtree(tmpdir)

def test_rotate_kv():
    im = Image.open('img_folder/opencv_logo.png')
    rgb_im = im # im.convert('RGB')
    r = Operations.Rotate(probability=1, rotation=-45)
    rgb_im = [rgb_im]
    rgb_im_r = r.perform_operation(rgb_im)
    rgb_im_r[0].save('img_folder/rotate.png')

#test_rotate_kv()

import cv2
from SimpleAugmentatations.ColorBrightness import  Color_Brightness

img = cv2.imread('img_folder/opencv_logo.png')
obj = Color_Brightness(img)
obj.get_brightness_scale()


