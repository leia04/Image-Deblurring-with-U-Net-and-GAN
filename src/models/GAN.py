#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, Activation, BatchNormalization, Add, Input,
    Conv2DTranspose, LeakyReLU, Flatten, Dense, Dropout, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16


class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        super().__init__(**kwargs)
        self.padding = tuple(padding)

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [w_pad, w_pad], [h_pad, h_pad], [0, 0]], mode="REFLECT")

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"padding": self.padding})
        return cfg


def res_block(x, filters, kernel_size=(3, 3), strides=(1, 1), use_dropout=False):
    y = x
    y = ReflectionPadding2D((1, 1))(y)
    y = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    if use_dropout:
        y = Dropout(0.5)(y)

    y = ReflectionPadding2D((1, 1))(y)
    y = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides)(y)
    y = BatchNormalization()(y)

    return Add()([x, y])


def generator_model(input_shape=(256, 256, 3)) -> tf.keras.Model:
    inputs = Input(shape=input_shape)

    x = ReflectionPadding2D((3, 3))(inputs)
    x = Conv2D(filters=64, kernel_size=(7, 7), padding="valid")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    n_downsampling = 2
    for i in range(n_downsampling):
        mult = 2 ** i
        x = Conv2D(filters=64 * mult * 2, kernel_size=(3, 3), strides=2, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

    mult = 2 ** n_downsampling
    for _ in range(9):
        x = res_block(x, 64 * mult, use_dropout=True)

    for i in range(n_downsampling):
        mult = 2 ** (n_downsampling - i)
        x = Conv2DTranspose(filters=int(64 * mult / 2), kernel_size=(3, 3), strides=2, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

    x = ReflectionPadding2D((3, 3))(x)
    x = Conv2D(filters=3, kernel_size=(7, 7), padding="valid")(x)
    x = Activation("tanh")(x)

    outputs = Add()([x, inputs])
    outputs = Lambda(lambda z: z / 2.0)(outputs)

    return Model(inputs=inputs, outputs=outputs, name="Generator")


def discriminator_model(input_shape=(256, 256, 3)) -> tf.keras.Model:
    inputs = Input(shape=input_shape)

    x = Conv2D(filters=64, kernel_size=(4, 4), strides=2, padding="same")(inputs)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=128, kernel_size=(4, 4), strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=256, kernel_size=(4, 4), strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=512, kernel_size=(4, 4), strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    outputs = Dense(1)(x)  # Wasserstein (no sigmoid)

    return Model(inputs=inputs, outputs=outputs, name="Discriminator")


# Losses
def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)


_vgg = None
def perceptual_loss(y_true, y_pred):
    """
    VGG16 feature-based perceptual loss.
    Assumes inputs are in [-1,1] or [0,1]. We map to [0,255] for VGG preprocessing.
    """
    global _vgg
    if _vgg is None:
        vgg = VGG16(include_top=False, weights="imagenet", input_shape=(256, 256, 3))
        vgg.trainable = False
        _vgg = Model(inputs=vgg.input, outputs=vgg.get_layer("block3_conv3").output)

    # to [0,1]
    yt = (y_true + 1.0) / 2.0 if tf.reduce_min(y_true) < 0 else y_true
    yp = (y_pred + 1.0) / 2.0 if tf.reduce_min(y_pred) < 0 else y_pred

    yt = tf.clip_by_value(yt, 0.0, 1.0) * 255.0
    yp = tf.clip_by_value(yp, 0.0, 1.0) * 255.0

    f_true = _vgg(yt)
    f_pred = _vgg(yp)
    return tf.reduce_mean(tf.abs(f_true - f_pred))
