from __future__ import division

from src.pipeline.dl_homograhpy.net_architecture import NetArchitecture
from src.pipeline.dl_homograhpy.net_config import *
import tensorflow as tf

class HomographyDL:
    """
    use a trained CNN-based model to recover homography

    This class is based on the approach described in:
    Recovering Homography from Camera Captured Documents using Convolutional Neural Networks (2017)

    """
    def __init__(self, input, output, architecture, model_fn, grayscale=False, pretrained_model=False):
        """Init"""
        self.input = input
        self.output = output
        self.architecture = architecture
        self.config = ArchitectureDefaultConfig
        self.model_fn = model_fn

        if architecture != None:
            self.model = NetArchitecture(architecture=architecture, config=self.config, output=self.output,
                                     grayscale=grayscale).create_cnn_model()
        else:
            self.model = None

        self.grayscale = grayscale
        self.pretrained_model = pretrained_model

    def evaluate_model(self, model, x_test, y_test):
        # evaluate model on test set
        score = model.evaluate(x_test, y_test, verbose=0)
        return score

    def import_model(self):
        """ Load model """
        return tf.keras.models.load_model(self.output + self.model_fn)

    def export_model(self):
        """ Save model to file """
        if self.grayscale:
            self.model.save(self.output + 'gray_' + self.model_fn)
        else:
            self.model.save(self.output + self.model_fn)

    def predict_corners(self, model, img):
        """
        Predict corners for image (img) using trained model (model)

        :param model:
        :param img:
        :return:
        """
        return model.predict(img)