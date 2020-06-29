import random
import cv2
import pandas as pd
import FileUtils.img_name_helper as im_helper
import FileUtils.img_roi_adapters as im_adapters
random.seed(18)

BasePath = '/home/shunya/Datasets/esad/train/set1'

def draw_pascal_rois_lbl(img, rois, lbls):
    color_roi = (255, 0, 0)
    color_font = (0, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    line_thickness = 2
    num_instance = rois.shape[0]
    roi_lbls = lbls.flatten()
    for ii in range(num_instance):
        pt1 = (int(rois[ii][0]), int(rois[ii][1]))
        pt2 = (int(rois[ii][2]), int(rois[ii][3]))
        pt3 = (int(rois[ii][0]) + 5, int(rois[ii][1]) + 5)
        lbl_str = str(int(roi_lbls[ii]))
        img = cv2.rectangle(img, pt1, pt2, color_roi, int(2))
        img = cv2.putText(img, lbl_str, pt3, font,
                    fontScale, color_font, line_thickness, cv2.LINE_AA)
    cv2.namedWindow("ImageRoi", cv2.WINDOW_NORMAL)
    cv2.imshow("ImageRoi", img)
    cv2.waitKey(0)

def parsefile2numpy(file_path):
    df_numpy = pd.read_csv(file_path, header=None, delim_whitespace=True).to_numpy()
    return df_numpy

def test_img_adapters():
    txt_path_list = im_helper.get_files_list(BasePath, 'txt')
    random.shuffle(txt_path_list)
    for ii in range(100):
        curr_txt_path = txt_path_list[ii]
        if not im_helper.is_file_empty(curr_txt_path):
            print(curr_txt_path)
            curr_img_path = curr_txt_path.replace('txt', 'jpg')
            img = cv2.imread(curr_img_path)
            data_np = parsefile2numpy(curr_txt_path)
            lbls = data_np[:, 0]
            rois = data_np[:, 1:]
            pascal_rois = im_adapters.yolo2pascal_rois(rois, img.shape)
            draw_pascal_rois_lbl(img, pascal_rois, lbls)



test_img_adapters()