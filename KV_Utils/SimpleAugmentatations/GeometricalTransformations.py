import cv2
import numpy as np

""""""
 #References
'https://blog.paperspace.com/data-augmentation-for-object-detection-rotation-and-shearing/'
""""""
class Geometrical_Transformation:
    def __init__(self, img):
        self.img = img

    def flip_image(self, type):
        if type == 'h':
            flip_img = cv2.flip(self.img, 1)
        elif type == 'v':
            flip_img = cv2.flip(self.img, 0)
        elif type == 'hv':
            flip_img = cv2.flip(self.img, -1)

        return flip_img

    def rotate_image_bound(self, angle):
        # grab the dimensions of the image and then determine the
        # centre
        (h, w) = self.img.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        rot_img = cv2.warpAffine(self.img, M, (nW, nH))

        return rot_img

    def rotate_roi(self, corners, angle):
        """"
        Rotate the bounding box
        Parameters
        __________
         corners : numpy.ndarray
        numpy.ndarray
        Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
        """""
        h, w, chnl = self.img.shape
        cx, cy = w / 2, h / 2
        corners = corners.reshape(-1, 2)
        corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))

        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy
        # Prepare the vector to be transformed
        calculated = np.dot(M, corners.T).T
        calculated = calculated.reshape(4, 2).astype(int)
        return calculated
