from unittest import TestCase
from src.pipeline.ocrMaster import OCRMaster
from src.pipeline.textline_recognition.ocr_config import *
import os

class TestOCRMaster(TestCase):

    input_data = '../../res/smartDocSamples/'
    output_data_mode0 = '../../res/smartDocSamplesOutput/mode0/'
    output_data_mode1 = '../../res/smartDocSamplesOutput/mode1/'

    # rgb p=1 xception
    homography_model = '../../res/homographyModel/xception_10000.h5'

    toc = TesseractOCRConfig

    # NOTE: to test images in portrait mode, set smartDoc=False (otherwise your images will be rotated by 90)
    # by default, all samples from the smartDoc2015 challenge 2 test set are oriented landscape

    #================================================================================#
    #                                   test Mode 0                                  #
    #================================================================================#

    '''
    # run Mode 0 on some samples from the SmartDoc2015 challenge 2 test set
    def test_run_ocr_pipe_model_0(self):

        # input, output, imgs, homography_model_fn, grayscale = True, mode=0, hocr=False, smartDoc=False
        ocr_master = OCRMaster(self.input_data, self.output_data_mode0, os.listdir(self.input_data),
                               homography_model_fn=None, 
                               grayscale=False, 
                               mode=0, 
                               hocr=False, 
                               smartDoc=True)
        ocr_master.run_ocr_pipe()
    '''

    #================================================================================#
    #                                   test Mode 1                                  #
    #================================================================================#

    # run Mode 1 on some samples from the SmartDoc2015 challenge 2 test set
    def test_run_ocr_pipe_model_1(self):

        # input, output, imgs, homography_model_fn, grayscale = True, mode=0, hocr=False, smartDoc=False
        ocr_master = OCRMaster(self.input_data, self.output_data_mode1, os.listdir(self.input_data),
                               homography_model_fn=self.homography_model,
                               grayscale=False,
                               mode=1,
                               hocr=False,
                               smartDoc=True)

        ocr_master.run_ocr_pipe()