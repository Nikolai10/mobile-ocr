class OCRBase:

    # abstract base class

    def __init__(self, ocr, intput, smartDoc=False):
        """init for all modes """
        self.ocr = ocr
        self.input = intput
        self.smartDoc = smartDoc

    def run(self, imgs):
        # override
        pass
