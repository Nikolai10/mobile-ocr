from __future__ import division
import cv2
import os

from src.pipeline.modes.OCRBase import OCRBase

class OCRMode0(OCRBase):
    """ OCR Mode 0: run tesseract only"""

    def run(self, imgs):
        for img_nm in sorted(imgs):
            print('Processing image {} ...'.format(img_nm))

            # load image
            input_img = self.input + img_nm
            img = cv2.imread(input_img)

            if self.smartDoc:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

            self.ocr.run_image_to_text_save(img, os.path.splitext(img_nm)[0])