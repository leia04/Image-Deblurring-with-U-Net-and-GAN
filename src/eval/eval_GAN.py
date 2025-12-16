#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from src.preprocessing.paired_dataset import load_image_pairs
from src.eval.metrics import compute_psnr, compute_ssim
from src.models.gan import ReflectionPadding2D


def eval_gan(
    model_path="artifacts/gan_generator.h5",
    test_root="data/GOPRO_Large/test",
    max_images=20,
    batch_size=1,
):
    generator = load_model(model_path, custom_objects={"ReflectionPadding2D": ReflectionPadding2D})
    ds = load_image_pairs(test_root, max_images=max_images, normalize="minus1_1").batch(batch_size)

    psnrs, ssims = [], []
    for blur, sharp in ds:
        pred = generator.predict(blur, verbose=0)
        psnrs.append(compute_psnr(sharp.numpy()[0], pred[0]))
        ssims.append(compute_ssim(sharp.numpy()[0], pred[0]))

    print(f"Mean PSNR: {float(np.mean(psnrs)):.4f}")
    print(f"Mean SSIM: {float(np.mean(ssims)):.4f}")


if __name__ == "__main__":
    eval_gan()

