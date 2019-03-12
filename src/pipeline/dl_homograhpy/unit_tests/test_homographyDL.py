from __future__ import division

from src.pipeline.dl_homograhpy.homographyDL import HomographyDL
from src.pipeline.dl_homograhpy.dl_homography_utils import *
from src.pipeline.dl_homograhpy.net_architecture import *
from tensorflow.python.client import device_lib
import numpy as np
from unittest import TestCase
import matplotlib.pyplot as plt
import tensorflow as tf


class TestHomographyDL(TestCase):

    input = '../../../../res/smartDocSamples/'
    output = ''

    test_img1 = '00001.jpg'
    test_img2 = '00002.jpg'
    test_img3 = '00003.jpg'
    test_img4 = '00004.jpg'
    test_img5 = '00005.jpg'

    # network spatial input shape
    input_shape = (384, 256)

    # rgb p=1 xception
    homography_model_fn = '../../../../res/homographyModel/xception_10000.h5'

    # input, output, architecture, model_fn
    homography_dl = HomographyDL(input=input,
                                 output=output,
                                 architecture='xception',
                                 model_fn=homography_model_fn,
                                 grayscale=False)

    # specify test object
    img = test_img3

    #----------------------------------------------------------------------------------------------------
    #                                           Network/ Config                                         #
    #----------------------------------------------------------------------------------------------------

    '''
    # Test network architecture
    def test_build_model(self):
       NetArchitecture(architecture='xception', config=ArchitectureDefaultConfig, output=self.output, grayscale=False)\
           .create_cnn_model()
    '''

    '''
    # Which Version; is GPU available?
    def test_tensorflow_config(self):
        print(tf.test.is_gpu_available)
        print(device_lib.list_local_devices())
    '''

    #----------------------------------------------------------------------------------------------------
    #                                            Inference                                              #
    #----------------------------------------------------------------------------------------------------

    # what does the trained model predict for a unseen image?
    def test_predict_and_visualize(self):

        # read rgb image
        img = cv2.imread(self.input + self.img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # manually rotate (should be automated) -> only if landscape images
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        # save original size
        org_y, org_x, _ = img.shape

        # resize (just for recovering homography)
        img_homography = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))

        # adjust dimension for network
        img_homography_net = np.reshape(img_homography, (1, self.input_shape[0], self.input_shape[1], 3))

        # normalize
        img_homography_norm = img_homography_net/255.0

        # estimate corner positions
        homography_model = self.homography_dl.import_model()
        corners = self.homography_dl.predict_corners(homography_model, img_homography_norm)

        # unwarp imgage (original size)
        pts_src = np.reshape(corners, (4, 2))
        pts_dst = np.array([[0, 0], [self.input_shape[1], 0], [self.input_shape[1], self.input_shape[0]],
                            [0, self.input_shape[0]]], dtype = 'float32')

        dewarped_image = warp_image(img_homography, pts_src, pts_dst, grayscale=False)
        print('Detected corners: {}'.format(pts_src))

        # visualize detected corners & dewarping result
        temp = visualize_xy(img_homography, corners[0])

        # plot
        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(temp)
        ax[1].imshow(dewarped_image)
        plt.show()