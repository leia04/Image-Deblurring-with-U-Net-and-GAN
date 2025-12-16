#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def preprocess_image(image_path, target_size=(256, 256)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


def postprocess_image(image_array):
    image_array = np.clip(image_array, 0, 1) * 255.0
    return image_array.astype(np.uint8)


def predict_unet(model_path, image_path):
    model = load_model(model_path)
    x = preprocess_image(image_path)
    y = model.predict(x, verbose=0)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Blurred")
    plt.imshow(x[0])

    plt.subplot(1, 2, 2)
    plt.title("Deblurred (U-Net)")
    plt.imshow(np.clip(y[0], 0, 1))
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # Edit paths as needed
    predict_unet("artifacts/unet.keras", "your_blur_image.png")
