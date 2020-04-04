import cv2
import numpy as np
class Color_Brightness:
    def __init__(self, img):
        self.img = img
    def get_brightness_scale(self):
        img_YCrCb = cv2.cvtColor(self.img, cv2.COLOR_BGR2YCrCb)
        y_chnl = img_YCrCb[:, :, 0].astype(float)
        y_chnl = y_chnl / 255.0
        score = round(np.mean(y_chnl), 3)
        return score

    def adjust_gamma(self, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        gamma_img = cv2.LUT(self.img, table)
        return gamma_img