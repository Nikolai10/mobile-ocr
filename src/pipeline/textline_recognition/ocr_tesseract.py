from __future__ import division

import pytesseract
import os
from ocr_config import *
from subprocess import call

class OcrTesseract:
    """
    this class uses the text line recognition module provided by
        - tesseract-ocr (https://github.com/tesseract-ocr/tesseract)
    using the pythonic wrapper
        - pytesseract (https://pypi.org/project/pytesseract/)

    to use pytesseract, install (https://github.com/tesseract-ocr/tesseract/wiki)

        sudo apt update
        sudo apt install tesseract-ocr
        sudo apt install libtesseract-dev

    to use other languages than english, install
    https://github.com/tesseract-ocr/tessdata_fast.git

    in /usr/share/tesseract-ocr/4.00/tessdata;
    (move content of tessdata_fast into tessdata)

    tesseract version: tesseract 4.0.0-beta.1 (LSTM version)
    tesseract config:

    --psm N
           Set Tesseract to only run a subset of layout analysis and assume a certain form of image. The options for N are:

               0 = Orientation and script detection (OSD) only.
               1 = Automatic page segmentation with OSD.
               2 = Automatic page segmentation, but no OSD, or OCR.
               3 = Fully automatic page segmentation, but no OSD. (Default)
               4 = Assume a single column of text of variable sizes.
               5 = Assume a single uniform block of vertically aligned text.
               6 = Assume a single uniform block of text.
               7 = Treat the image as a single text line.
               8 = Treat the image as a single word.
               9 = Treat the image as a single word in a circle.
               10 = Treat the image as a single character.

       --oem N
           Specify OCR Engine mode. The options for N are:

               0 = Original Tesseract only.
               1 = Neural nets LSTM only.
               2 = Tesseract + LSTM.
               3 = Default, based on what is available.

    for more information, please check the Manual provided by tesseract (cmd tool)

    """
    def __init__(self, input, output):
        """
        setup OCR Engine

        :param input:
        :param output:
        :param debug:
        """
        self.input = input              # input dir
        self.output = output            # output dir
        self.toc = TesseractOCRConfig   # load global Configs

    def run_image_to_text(self, img):
        """
        run OCR on one image

        :param img:     cv2 img
        :return:
        """

        # build config and run
        config = self.toc.psm_param + ' ' + self.toc.psm + ' ' + self.toc.oem_param + ' ' + self.toc.oem
        return pytesseract.image_to_string(img, lang=self.toc.lang, config=config)

    def run_image_to_text_save(self, img, img_fn):
        """
        run_image_to_text + save to file

        :param img:
        :param img_fn:
        :return:
        """
        # retrieve textline_recognition result
        ocr_result = self.run_image_to_text(img)

        # create output file
        output_file = self.output + img_fn + '.' + 'txt'

        with open(output_file, 'w') as out_file:
            out_file.write(ocr_result.encode('utf-8'))

    # deprecated (specify config file, with param: tessedit_create_hocr 1)
    def layout_analysis_to_hocr(self, img):
        """retrieve HOCR file from image (HTML OCR format)"""

        img_path = self.input + img
        output_file = self.output + os.path.splitext(img)[0]

        #tesseract input.png output --psm 3 --oem 1 hocr
        call(['tesseract', img_path, output_file, self.toc.psm_param, self.toc.psm,
              self.toc.oem_param, self.toc.oem, self.toc.hocr_format])

    def scale_img(self, img):
        """scale image to 1000pxls height"""
        h = img.shape[0]

        scale = 1
        if h > self.toc.max_h:
            scale = self.toc.max_h/h

        return cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
