#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from src.preprocessing.paired_dataset import load_image_pairs
from src.eval.metrics import compute_psnr, compute_ssim


def eval_unet(
    model_path="artifacts/unet.keras",
    test_root="data/GOPRO_Large/test",
    max_images=20,
    batch_size=1,
):
    model = load_model(model_path)
    ds = load_image_pairs(test_root, max_images=max_images, normalize="0_1").batch(batch_size)

    psnrs, ssims = [], []
    for blur, sharp in ds:
        pred = model.predict(blur, verbose=0)
        psnrs.append(compute_psnr(sharp.numpy()[0], pred[0]))
        ssims.append(compute_ssim(sharp.numpy()[0], pred[0]))

    print(f"Mean PSNR: {float(np.mean(psnrs)):.4f}")
    print(f"Mean SSIM: {float(np.mean(ssims)):.4f}")


if __name__ == "__main__":
    eval_unet()
