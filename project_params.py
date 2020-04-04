import os
import time
import sys
from kv_utils.file_utils import parse_experiment_params, img_name_helper

cfg_file_name = 'params.cfg'
cfg_file_path = os.path.join(os.path.dirname(__file__), cfg_file_name)
print('### WARNING: Make sure name of the CFG file is correct ###')
print('### Name of CFG File ----> ', cfg_file_path)
time.sleep(2)

parse_exp_path_ob = parse_experiment_params.ParseExpParams(cfg_file_path)
##Project params
Project_Name = parse_exp_path_ob.get_project_name()
Unstructred_Data_Path = parse_exp_path_ob.get_unstructured_data_path()
Base_Path = parse_exp_path_ob.get_base_data_path()
exp_checkpoint_path = parse_exp_path_ob.get_checkpoints_path()
exp_cyclic_policy = parse_exp_path_ob.get_cyclic_lr_policy()

##create checkpoints
#restore_point = 1 #int(sys.argv[2])
#if restore_point==0:
img_name_helper.create_clean_dirs(exp_checkpoint_path)

img_ext = parse_exp_path_ob.img_ext
num_classes = parse_exp_path_ob.get_num_classes()

#HyperParams
exp_learning_rate = parse_exp_path_ob.get_learning_rate()
exp_batch_sz = parse_exp_path_ob.get_batch_size()
exp_validation_frac = parse_exp_path_ob.get_validation_fraction()
img_size = parse_exp_path_ob.get_img_dim()
img_channels = parse_exp_path_ob.get_img_channels()
input_shape = (img_size, img_size, img_channels)
exp_max_epochs = parse_exp_path_ob.get_max_no_epochs()
