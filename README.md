# MobileOCR
Keywords: `OCR`, `Page Dewarping`, `Deep Learning`, `Tesseract-OCR`, `DAS2018`, `ICDAR2017` 

[Conference Paper](https://conferences.ulbsibiu.ro/icdd/2019/files/Proceedings_ICDD2019.pdf)

## 1. Introduction: Camera-based document analysis

This repository is part of the Master Thesis "Camera-based Document Analysis based on Deep Learning and OCR". 

Capturing document images with the Smartphone provide a convenient way to digitize physical documents and facilitate the 
automation of document processing and information retrieval. In contrast to flatbed scans, camera-captured documents 
require a more sophisticated preprocessing pipeline, because of perspective distortions, suboptimal lighting and 
physically deformed documents. The main goal of this work was to:
- build an end-to-end OCR-Pipeline (input: document image, output: full text transcription) based on the best 
Open Source solution currently available. 
- analyze Deep Learning techniques to deal with one of the major challenges discussed at the DAS2018 
workshop in the domain of camera-based document analysis: Page Dewarping (in particular perspective distortions 
and folded/ curved documents).

 A high-level overview is illustrated in the following figure:

<p align="center">
  <img width="70%" height="70%" src=doc/EvalFramework.png>
</p>

This repo contains a demo application for the by the page dewarping component extended OCR pipeline 
([Tesseract 4.0](https://github.com/tesseract-ocr/tesseract)).

## 2. Proposed Method

Methodically, different neural network architectures were investigated on a large-scale synthetic dataset to estimate 
the document's corner points from a single input image, without prior assumptions. The distorted image is then mapped 
to its canonical position by using the 4-point homography parameterization. The best result is achieved by a modified 
[Xception-network](https://arxiv.org/pdf/1610.02357.pdf), with a mean displacement error of 3.38px. 
Finally, the correction component is integrated into Tesseract 4.0 and evaluated on the [SmartDoc 2015 challenge 2 test 
set](http://www.cvc.uab.es/~marcal/pdfs/ICDAR15e.pdf). Experiments show that the correction component improves 
the character accuracy results by more than 15 percentage points (93.11\%), in comparison to Tesseract alone (77.27\%).  

## 3. Demo Examples

Page Dewarping results:

<p align="center">
  <img width="70%" height="70%" src=doc/pageDewarping.png>
</p>

By tesseract recognized textlines before after dewarping:

<p align="center">
  <img width="70%" height="70%" src=doc/recognized_textlines.png>
</p>

## 4. Setup

1. [Install Tesseract OCR](https://github.com/tesseract-ocr/tesseract); at time of writing, tesseract 4.0.0-beta.1 
was used as OCR engine.

2. Download [homography_model](https://www.dropbox.com/s/8ywndwenfedjab8/xception_10000.h5?dl=0) into /res/homographyModel/

3. Install dependencies (using conda virtualenv)
    
```     
    conda env create -f environment.yml
    # note: to use gpu support, exchange tensorflow with tensorflow-gpu (environment.yml)
```

## 5. Usage

To test different pipeline modes, consider [ocrMaster](src/pipeline/test_OCRMaster.py); to test the page dewarping 
performance [test_homographyDL.py](src/pipeline/dl_homograhpy/unit_tests/test_homographyDL.py).

## 6. File Structure

    doc
        ├── ...                                                          # README resources
    res                               
        ├── homographyModel/                                             # dir to trained homography model 
        ├── smartDocSamples/                                             # smartDoc challenge 2 test set samples
        ├── smartDocSamplesOutput/                                       # mobileOCR results are stored here   
    src
        ├── pipeline/                                                    # mobileOCR pipeline
            ├── dl_homograhpy_tf/                                        # deep learning: homography; dewarping 
            ├── modes/                                                   # different OCR pipieline modes
            ├── textline_recognition/                                    # Tesseract 4.0 python wrapper 
            ├── ocrMaster.py                                             # interface to OCR pipeline
    environment.yml                                                      # dependencies (easy setup)

## License

[mobileOCR_License](LICENSE)
