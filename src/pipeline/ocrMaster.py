from src.pipeline.textline_recognition.ocr_tesseract import OcrTesseract
from modes.OCRMode0 import OCRMode0
from modes.OCRMode1 import OCRMode1
import tensorflow as tf

import logging

class OCRMaster:
    """
    OCR master integrate sub-processes (slaves) of OCR-Pipeline into one master-node
    (controller)

    config:
        - mode: 0               tesseract only
        - mode: 1               DL Homography + tesseract
    """

    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    logger = logging.getLogger('OCR Master')

    def __init__(self, input, output, imgs, homography_model_fn, grayscale = True, mode=0, hocr=False, smartDoc=False):
        """
        Init Master

        :param input:                   input dir
        :param output:                  output dir
        :param imgs:                    list of images to process
        :param homography_model_fn:     path to trained model
        :param grayscale:               rgb vs. grayscale
        :param mode:                    textline_recognition mode
        :param hocr:                    whether to create hocr file (layout analysis)
        :param smartDoc:                rotate if smartDoc dataset (landscape images)
        """
        self.input = input
        self.output = output
        self.mode = mode
        self.imgs = imgs
        self.homography_model_fn = homography_model_fn
        self.grayscale = grayscale
        self.hocr = hocr
        self.smartDoc = smartDoc

        # build OCR pipe
        self.build_ocr_pipe()

    def build_ocr_pipe(self):
        """build textline_recognition pipe"""
        self.ocr = OcrTesseract(self.input, self.output)
        if self.homography_model_fn is not None:
            self.homography_model = tf.keras.models.load_model(self.homography_model_fn)

    #================================================================================#
    #                               Run Pipe on dataset                              #
    #================================================================================#

    def run_ocr_pipe(self):
        """run textline_recognition pipe on a set of images; store results as txt files"""
        self.logger.info('Run mode {}'.format(self.mode))
        if self.mode == 0:
            OCRMode0(self.ocr, self.input, self.smartDoc).run(self.imgs)
        elif self.mode == 1:
            OCRMode1(self.ocr, self.input, self.smartDoc, self.homography_model, self.grayscale).run(self.imgs)
        else:
            raise ValueError('mode not supported')