import cv2
import numpy as np
class Noise_Transformations:
    def __init__(self, img):
        self.img = img

    def noisy_img(self, noise_type, alpha=0.5):
        if noise_type == "gauss":
            row, col, ch = self.img.shape
            mean = 0
            var = 0.1
            sigma = var**0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            gauss = (gauss - np.min(gauss))/np.ptp(gauss)
            noisy = alpha * (self.img / 255.0) + (1 - alpha) * gauss
            noisy = (noisy * 255.0).astype(dtype=np.uint8)
            return noisy
        elif noise_type == "sp":
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(self.img)
            # Salt mode
            num_salt = np.ceil(amount * self.img.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in self.img.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * self.img.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in self.img.shape]
            out[coords] = 0
            return out
        elif noise_type == "poisson":
            vals = len(np.unique(self.img))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(self.img * vals) / float(vals)
            return noisy
        elif noise_type =="speckle":
            row, col, ch = self.img.shape
            gauss = np.random.randn(row,col,ch)
            gauss = gauss.reshape(row,col,ch)
            noisy = self.img + self.img * gauss
        return noisy