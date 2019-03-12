"""pipeline config"""

class TesseractOCRConfig:

    # output format (HTML OCR)
    hocr_format = 'hocr'
    # specify layout analysis
    psm = '3'
    psm_param = '--psm'
    # specify OCR Engine mode
    oem = '1'
    oem_param = '--oem'
    # specify languages
    lang='eng'