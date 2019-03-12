from unittest import TestCase
from src.pipeline.textline_recognition.ocr_tesseract import OcrTesseract
from src.pipeline.textline_recognition.ocr_config import *
import cv2

class TestTesseractOCR(TestCase):

    input = '../../../../res/smartDocSamples/'
    output = ''

    test_img1 = '00001.jpg'
    test_img2 = '00002.jpg'
    test_img3 = '00003.jpg'
    test_img4 = '00004.jpg'
    test_img5 = '00005.jpg'

    toc = TesseractOCRConfig

    # specify test object
    img = test_img1

    #================================================================================#
    #                               test one sample                                  #
    #================================================================================#

    def test_image_to_text(self):
        ocr = OcrTesseract(self.input, self.output)
        img = cv2.imread(self.input + self.img)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        print(ocr.run_image_to_text(img))