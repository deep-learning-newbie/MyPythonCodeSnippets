import os
import glob
import shutil
import numpy as np
import cv2

def get_path_name_frm_index(img_index, out_dir):
    img_name = "%6.6d.png" % (img_index)
    img_path = os.path.join(out_dir, img_name)
    return img_path, img_name

def get_filename_noext(file_name):
    file_dir, file_name = os.path.split(file_name)
    file_name_no_ext, file_ext = os.path.splitext(file_name)
    return file_name_no_ext

def get_files_list(in_dir, ext, is_recursive = True, is_srt=False):
    list_files = []
    if is_recursive:
        src_dir = in_dir + '/**/*.' + ext
        list_files = glob.glob(src_dir, recursive=True)
    else:
        src_dir = in_dir + '/*.' + ext
        list_files = glob.glob(src_dir)

    if is_srt:
        list_files.sort()

    return list_files

def create_clean_dirs(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def is_file_empty(file_path):
    return os.stat(file_path).st_size == 0
