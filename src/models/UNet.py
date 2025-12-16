#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Input
from tensorflow.keras.models import Model


def unet_model(input_size=(256, 256, 3)) -> tf.keras.Model:
    inputs = Input(input_size)

    def conv_block(x, filters):
        x = Conv2D(filters, 3, activation="relu", padding="same")(x)
        x = Conv2D(filters, 3, activation="relu", padding="same")(x)
        return x

    def encoder_block(x, filters):
        s = conv_block(x, filters)
        p = MaxPooling2D((2, 2))(s)
        return s, p

    def decoder_block(x, skip_features, filters):
        x = Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(x)
        x = concatenate([x, skip_features])
        x = conv_block(x, filters)
        return x

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b = conv_block(p4, 1024)

    d4 = decoder_block(b, s4, 512)
    d3 = decoder_block(d4, s3, 256)
    d2 = decoder_block(d3, s2, 128)
    d1 = decoder_block(d2, s1, 64)

    outputs = Conv2D(3, 1, activation="linear")(d1)
    return Model(inputs, outputs, name="U-Net")
