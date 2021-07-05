import os
import urllib

import cv2
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import GlobalAveragePooling2D, multiply, Permute
from keras.layers import Input, Dense, Reshape, Dropout, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from skimage import transform as trans

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = BASE_DIR + "/checkpoints/baldgan/"

model_paths = {
    # 'D': model_path + 'model_D_5_170.hdf5',
    # 'D_mask': model_path + 'model_D_mask_5_170.hdf5',
    'G': model_path + 'model_G_5_170.hdf5'
}


class BaldGAN:
    def __init__(self):
        if not os.path.isfile(model_paths['G']):
            print("baldGAN : failed to find model, downloading...")
            url = 'https://jinwoo17962.synology.me/datasets/baldgan/model_G_5_170.hdf5'
            urllib.request.urlretrieve(url, model_paths['G'])
        K.set_learning_phase(0)

        # Image input
        d0 = Input(shape=(256, 256, 3))

        gf = 64
        # Downsampling
        d1 = conv2d(d0, gf, bn=False, se=True)
        d2 = conv2d(d1, gf * 2, se=True)
        d3 = conv2d(d2, gf * 4, se=True)
        d4 = conv2d(d3, gf * 8)
        d5 = conv2d(d4, gf * 8)

        a1 = atrous(d5, gf * 8)

        # Upsampling
        u3 = deconv2d(a1, d4, gf * 8)
        u4 = deconv2d(u3, d3, gf * 4)
        u5 = deconv2d(u4, d2, gf * 2)
        u6 = deconv2d(u5, d1, gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(3, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        self.model = Model(d0, output_img)
        self.model.load_weights(model_paths['G'])

    def go_bald(self, image: np.ndarray):
        input_face = np.expand_dims(image, axis=0)
        input_face = (input_face / 127.5) - 1.
        result = self.model.predict(input_face)[0]
        result = ((result + 1.) * 127.5)
        result = result.astype(np.uint8)
        return result


def squeeze_excite_block(input, ratio=4):
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def conv2d(layer_input, filters, f_size=4, bn=True, se=False):
    d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    if bn:
        d = InstanceNormalization()(d)
    if se:
        d = squeeze_excite_block(d)
    return d


def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
    u = UpSampling2D(size=2)(layer_input)
    u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
    if dropout_rate:
        u = Dropout(dropout_rate)(u)
    u = InstanceNormalization()(u)
    u = Concatenate()([u, skip_input])
    return u


def atrous(layer_input, filters, f_size=4, bn=True):
    a_list = []

    for rate in [2, 4, 8]:
        # a = AtrousConvolution2D(filters, f_size, atrous_rate=rate, border_mode='same')(layer_input)
        a = Conv2D(filters, kernel_size=f_size, dilation_rate=rate, padding='same')(layer_input)
        a_list.append(a)
    a = Concatenate()(a_list)
    a = LeakyReLU(alpha=0.2)(a)
    if bn:
        a = InstanceNormalization()(a)
    return a
