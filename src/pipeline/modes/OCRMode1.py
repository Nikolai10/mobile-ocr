from src.pipeline.modes.OCRBase import OCRBase
from src.pipeline.dl_homograhpy.homographyDL import HomographyDL
import numpy as np
import cv2
import os
from src.pipeline.dl_homograhpy.dl_homography_utils import *

class OCRMode1(OCRBase):
    """ OCR Mode 1: DL Homography + tesseract"""

    def __init__(self, ocr, intput, smartDoc, homography_model, grayscale):
        OCRBase.__init__(self, ocr, intput, smartDoc)
        self.homography_model = homography_model
        self.grayscale = grayscale

    # network spatial input shape
    input_shape = (384, 256)

    # create empty instance
    homography_dl = HomographyDL(input=None, output=None, architecture=None, model_fn=None, grayscale=None)

    def run(self, imgs):
        for img_nm in sorted(imgs):
            print('Processing image {} ...'.format(img_nm))

            # load image
            input_img = self.input + img_nm
            if self.grayscale:
                img = cv2.imread(input_img, 0)
            else:
                img = cv2.imread(input_img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # manually rotate (should be automated)
            if self.smartDoc:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

            # save original size to compute scaling factor
            if self.grayscale:
                org_y, org_x = img.shape
            else:
                org_y, org_x, _ = img.shape

            fac_y, fac_x = org_y/self.input_shape[0], org_x/self.input_shape[1]

            # resize (just for recovering homography)
            img_homography = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))

            # adjust dimension for network
            if self.grayscale:
                img_homography_net = np.reshape(img_homography, (1, self.input_shape[0], self.input_shape[1], 1))
            else:
                img_homography_net = np.reshape(img_homography, (1, self.input_shape[0], self.input_shape[1], 3))

            # normalize
            img_homography_norm = img_homography_net/255.0

            # estimate corner positions
            corners = self.homography_dl.predict_corners(self.homography_model, img_homography_norm)

            # unwarp imgage (original size)
            pts_src = np.reshape(corners, (4, 2))
            pts_src = self.scale_estim_corners(pts_src, fac_x, fac_y)
            pts_dst = np.array([[0, 0], [org_x, 0], [org_x, org_y], [0, org_y]], dtype = 'float32')

            dewarped_image = warp_image(img, pts_src, pts_dst, self.grayscale)

            # tesseract
            self.ocr.run_image_to_text_save(dewarped_image, os.path.splitext(img_nm)[0])

    def scale_estim_corners(self, corners, scale_x, scale_y):
        """
        scale estimated corners to original image size

        :param corners:
        :param scale_x:
        :param scale_y:
        :return:
        """
        erg = np.zeros((4,2))

        for idx, corner_tuple in enumerate(corners):
            erg[idx] = corner_tuple[0]*scale_x,corner_tuple[1]*scale_y

        return erg