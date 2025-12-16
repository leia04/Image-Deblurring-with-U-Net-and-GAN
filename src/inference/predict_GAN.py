#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from src.models.gan import ReflectionPadding2D


def load_and_preprocess_image(image_path, target_size=(256, 256)):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img).astype(np.float32) / 255.0
    img = img * 2.0 - 1.0  # [-1,1] for GAN
    return np.expand_dims(img, axis=0)


def postprocess_image(image):
    image = image[0]
    image = (image + 1.0) / 2.0
    image = np.clip(image, 0, 1)
    return (image * 255).astype(np.uint8)


def predict_gan(model_path, image_path):
    generator = load_model(model_path, custom_objects={"ReflectionPadding2D": ReflectionPadding2D})

    blur = load_and_preprocess_image(image_path)
    pred = generator.predict(blur, verbose=0)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Blurred")
    plt.imshow((blur[0] + 1) / 2.0)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Deblurred (GAN)")
    plt.imshow(postprocess_image(pred))
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # Edit paths as needed
    predict_gan("artifacts/gan_generator.h5", "your_blur_image.png")
