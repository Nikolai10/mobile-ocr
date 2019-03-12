from __future__ import print_function, division

from src.pipeline.dl_homograhpy.net_config import *
import tensorflow as tf
from tensorflow import keras


class NetArchitecture:

    """
    NetArchitecture returns a variety of different neural network architectures depending on CNNs

    The 'NUCES-FAST' architecture is based on:
    Recovering Homography from Camera Captured Documents using Convolutional Neural Networks (2017)

    architecture overview:
        - default:          NUCES-FAST + batch norm after each Conv2D
        - nucesfast:        paper:  https://arxiv.org/pdf/1709.03524.pdf;
        - xception:         paper:  https://arxiv.org/pdf/1409.4842.pdf;

        Xception implementation based on:
        https://github.com/keras-team/keras-applications/blob/master/keras_applications/xception.py


    Note: applying batch norm before or after a RELU is still a controversial topic; in this work, the following order
    is chosen: Conv2d -> Relu -> Batch Norm -> Dropout

    ref:
        - https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout-in-tensorflow
        - https://github.com/keras-team/keras/issues/1802#issuecomment-187966878
    """

    def __init__(self, architecture, config, output, grayscale):
        """Init params"""
        self.architecture = architecture
        self.config = config
        self.output = output
        self.grayscale = grayscale

    def create_cnn_model(self):
        """ define model architecture and compile """

        input = self.config.input_shape_rgb if not self.grayscale else self.config.input_shape_gray
        # default
        if self.architecture == 'default':
            model = keras.Sequential([
                #============================================= 1st layer ==============================================#
                keras.layers.Conv2D(64, kernel_size=(5, 5), padding='same', strides=(1, 1),
                                    kernel_initializer = 'he_normal', activation='relu', input_shape=input),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
                #============================================= 2nd layer ==============================================#
                keras.layers.Conv2D(128, kernel_size=(5, 5), padding='same', strides=(1, 1), activation='relu',
                                    kernel_initializer = 'he_normal'),
                keras.layers.BatchNormalization(),
                #============================================= 3rd layer ==============================================#
                keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu',
                                    kernel_initializer = 'he_normal'),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
                #============================================= 4th layer ==============================================#
                keras.layers.Conv2D(384, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu',
                                    kernel_initializer = 'he_normal'),
                keras.layers.BatchNormalization(),
                #============================================= 5th layer ==============================================#
                keras.layers.Conv2D(384, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu',
                                    kernel_initializer = 'he_normal'),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
                #============================================= 6th layer ==============================================#
                keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu',
                                    kernel_initializer = 'he_normal'),
                keras.layers.BatchNormalization(),
                #============================================= 7th layer ==============================================#
                keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu',
                                    kernel_initializer = 'he_normal'),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
                #============================================= 8th layer ==============================================#
                keras.layers.Conv2D(1024, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu',
                                    kernel_initializer = 'he_normal'),
                keras.layers.BatchNormalization(),
                #============================================= 9th layer ==============================================#
                keras.layers.Conv2D(1024, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu',
                                    kernel_initializer = 'he_normal'),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
                #=============================================10th layer ==============================================#
                keras.layers.Conv2D(1024, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu',
                                    kernel_initializer = 'he_normal'),
                keras.layers.BatchNormalization(),
                #=============================================11th layer ==============================================#
                keras.layers.Conv2D(2048, kernel_size=(1, 1), padding='same', strides=(1, 1),
                                    kernel_initializer = 'he_normal'),
                keras.layers.BatchNormalization(),

                # FCN
                keras.layers.Dropout(0.5),
                keras.layers.Flatten(),
                keras.layers.Dense(8, kernel_initializer = 'he_normal')
            ])

        elif self.architecture == 'nucesfast':
            model = keras.Sequential([
                #============================================= 1st layer ==============================================#
                keras.layers.Conv2D(64, kernel_size=(5, 5), padding='same', strides=(1, 1),
                                    kernel_initializer = 'he_normal', activation='relu', input_shape=input),
                keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
                #============================================= 2nd layer ==============================================#
                keras.layers.Conv2D(128, kernel_size=(5, 5), padding='same', strides=(1, 1), activation='relu',
                                    kernel_initializer = 'he_normal'),
                #============================================= 3rd layer ==============================================#
                keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu',
                                    kernel_initializer = 'he_normal'),
                keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
                #============================================= 4th layer ==============================================#
                keras.layers.Conv2D(384, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu',
                                    kernel_initializer = 'he_normal'),
                #============================================= 5th layer ==============================================#
                keras.layers.Conv2D(384, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu',
                                    kernel_initializer = 'he_normal'),
                keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
                #============================================= 6th layer ==============================================#
                keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu',
                                    kernel_initializer = 'he_normal'),
                #============================================= 7th layer ==============================================#
                keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu',
                                    kernel_initializer = 'he_normal'),
                keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
                #============================================= 8th layer ==============================================#
                keras.layers.Conv2D(1024, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu',
                                    kernel_initializer = 'he_normal'),
                #============================================= 9th layer ==============================================#
                keras.layers.Conv2D(1024, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu',
                                    kernel_initializer = 'he_normal'),
                keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
                #=============================================10th layer ==============================================#
                keras.layers.Conv2D(1024, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu',
                                    kernel_initializer = 'he_normal'),
                #=============================================11th layer ==============================================#
                keras.layers.Conv2D(2048, kernel_size=(1, 1), padding='same', strides=(1, 1),
                                    kernel_initializer = 'he_normal'),
                # FCN
                keras.layers.Dropout(0.5),
                keras.layers.Flatten(),
                keras.layers.Dense(8, kernel_initializer = 'he_normal')
            ])

        elif self.architecture == 'xception':
            # placeholder tensor
            inputs = tf.keras.Input(shape=input)
            #=============================================== 1st Block ================================================#
            x = keras.layers.Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(inputs)
            x = keras.layers.BatchNormalization(name='block1_conv1_bn')(x)
            x = keras.layers.Activation('relu', name='block1_conv1_act')(x)
            x = keras.layers.Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
            x = keras.layers.BatchNormalization(name='block1_conv2_bn')(x)
            x = keras.layers.Activation('relu', name='block1_conv2_act')(x)
            # residual
            residual = keras.layers.Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
            residual = keras.layers.BatchNormalization()(residual)
            #=============================================== 2nd Block ================================================#
            x = keras.layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
            x = keras.layers.BatchNormalization(name='block2_sepconv1_bn')(x)
            x = keras.layers.Activation('relu', name='block2_sepconv2_act')(x)
            x = keras.layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
            x = keras.layers.BatchNormalization(name='block2_sepconv2_bn')(x)
            # pooling
            x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
            x = keras.layers.add([x, residual])
            # residual
            residual = keras.layers.Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
            residual = keras.layers.BatchNormalization()(residual)
            #=============================================== 3rd Block ================================================#
            x = keras.layers.Activation('relu', name='block3_sepconv1_act')(x)
            x = keras.layers.SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
            x = keras.layers.BatchNormalization(name='block3_sepconv1_bn')(x)
            x = keras.layers.Activation('relu', name='block3_sepconv2_act')(x)
            x = keras.layers.SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
            x = keras.layers.BatchNormalization(name='block3_sepconv2_bn')(x)
            # pooling
            x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
            x = keras.layers.add([x, residual])
            # residual
            residual = keras.layers.Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
            residual = keras.layers.BatchNormalization()(residual)
            #=============================================== 4rd Block ================================================#
            x = keras.layers.Activation('relu', name='block4_sepconv1_act')(x)
            x = keras.layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
            x = keras.layers.BatchNormalization(name='block4_sepconv1_bn')(x)
            x = keras.layers.Activation('relu', name='block4_sepconv2_act')(x)
            x = keras.layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
            x = keras.layers.BatchNormalization(name='block4_sepconv2_bn')(x)
            # pooling
            x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
            x = keras.layers.add([x, residual])
            #=============================================== 5-12th Block =============================================#
            for i in range(8):
                residual = x
                prefix = 'block' + str(i + 5)
                x = keras.layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
                x = keras.layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
                x = keras.layers.BatchNormalization(name=prefix + '_sepconv1_bn')(x)
                x = keras.layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
                x = keras.layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
                x = keras.layers.BatchNormalization(name=prefix + '_sepconv2_bn')(x)
                x = keras.layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
                x = keras.layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
                x = keras.layers.BatchNormalization(name=prefix + '_sepconv3_bn')(x)
                x = keras.layers.add([x, residual])
            # residual
            residual = keras.layers.Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
            residual = keras.layers.BatchNormalization()(residual)
            #=============================================== 13th Block ===============================================#
            x = keras.layers.Activation('relu', name='block13_sepconv1_act')(x)
            x = keras.layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
            x = keras.layers.BatchNormalization(name='block13_sepconv1_bn')(x)
            x = keras.layers.Activation('relu', name='block13_sepconv2_act')(x)
            x = keras.layers.SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
            x = keras.layers.BatchNormalization(name='block13_sepconv2_bn')(x)
            # pooling
            x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
            x = keras.layers.add([x, residual])
            #=============================================== 14th Block ===============================================#
            x = keras.layers.SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
            x = keras.layers.BatchNormalization(name='block14_sepconv1_bn')(x)
            x = keras.layers.Activation('relu', name='block14_sepconv1_act')(x)
            x = keras.layers.SeparableConv2D(2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
            x = keras.layers.BatchNormalization(name='block14_sepconv2_bn')(x)
            x = keras.layers.Activation('relu', name='block14_sepconv2_act')(x)
            #============================================= custom block ===============================================#
            x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
            x = keras.layers.Dense(8, name='predictions')(x)
            # Create model.
            model = keras.models.Model(inputs, x, name='xception')
        else:
            raise TypeError('Either: {} is supported'.format('default, xception or nucesfast'))

        # compile network
        model.compile(loss=self.config.loss, optimizer=self.config.optimizer)
        model.summary()
        #tf.keras.utils.plot_model(model, to_file=self.output + self.architecture + '.png', show_shapes=True)
        return model