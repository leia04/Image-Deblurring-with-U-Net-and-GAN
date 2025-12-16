#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array


IMG_EXT = (".png", ".jpg", ".jpeg")


def _load_images_from_subdir(subdir: str, max_images: int | None, target_size=(256, 256)):
    images = []
    count = 0
    for file in os.listdir(subdir):
        if file.lower().endswith(IMG_EXT):
            img_path = os.path.join(subdir, file)
            img = load_img(img_path, target_size=target_size)
            img = img_to_array(img).astype(np.float32) / 255.0  # [0,1]
            images.append(img)
            count += 1
            if max_images and count >= max_images:
                break
    return images


def load_image_pairs(root_dir: str, max_images: int | None = None, target_size=(256, 256), normalize: str = "0_1"):
    """
    Loads paired (blur, sharp) images under a root directory.
    Expected structure includes folders containing 'blur' and 'sharp' in path names.
    Returns a tf.data.Dataset of (blur, sharp).

    normalize:
      - "0_1": keep in [0,1] (U-Net)
      - "minus1_1": scale to [-1,1] (GAN)
    """
    blur_images = []
    sharp_images = []

    blur_count = 0
    sharp_count = 0

    for subdir, _, files in os.walk(root_dir):
        if "blur" in subdir:
            for file in files:
                if file.lower().endswith(IMG_EXT):
                    img_path = os.path.join(subdir, file)
                    img = load_img(img_path, target_size=target_size)
                    img = img_to_array(img).astype(np.float32) / 255.0
                    blur_images.append(img)
                    blur_count += 1
                    if max_images and blur_count >= max_images:
                        break

        elif "sharp" in subdir:
            for file in files:
                if file.lower().endswith(IMG_EXT):
                    img_path = os.path.join(subdir, file)
                    img = load_img(img_path, target_size=target_size)
                    img = img_to_array(img).astype(np.float32) / 255.0
                    sharp_images.append(img)
                    sharp_count += 1
                    if max_images and sharp_count >= max_images:
                        break

        if max_images and (blur_count >= max_images and sharp_count >= max_images):
            break

    min_count = min(len(blur_images), len(sharp_images))
    blur_images = blur_images[:min_count]
    sharp_images = sharp_images[:min_count]

    blur = tf.convert_to_tensor(blur_images, dtype=tf.float32)
    sharp = tf.convert_to_tensor(sharp_images, dtype=tf.float32)

    if normalize == "minus1_1":
        blur = blur * 2.0 - 1.0
        sharp = sharp * 2.0 - 1.0

    return tf.data.Dataset.from_tensor_slices((blur, sharp))
