import os
import glob
import numpy as np
import cv2

def get_img_path_name_frm_index(img_index, out_dir):
    img_name = "%6.6d.png" % (img_index)
    img_path = os.path.join(out_dir, img_name)
    return img_path, img_name

def get_file_number(file_name):
    file_dir, file_name = os.path.split(file_name)
    file_no, file_ext = os.path.splitext(file_name)
    return int(file_no)

def get_filename_noext(file_name):
    file_dir, file_name = os.path.split(file_name)
    file_name_no_ext, file_ext = os.path.splitext(file_name)
    return file_name_no_ext

def get_img_list(in_dir, ext, is_srt=False):
    #src_dir = in_dir + '/*.' + ext
    src_dir = in_dir + '/**/*.' + ext
    list_images = glob.glob(src_dir, recursive=True)
    if is_srt:
        list_images.sort()
    return list_images

def get_files_list(in_dir, ext, is_srt=False):
    src_dir = in_dir + '/*.' + ext
    list_files = glob.glob(src_dir)
    if is_srt:
        list_files.sort()
    return list_files