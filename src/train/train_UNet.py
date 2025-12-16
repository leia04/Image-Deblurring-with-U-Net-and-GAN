#!/usr/bin/env python
# coding: utf-8

import os
import tensorflow as tf

from src.preprocessing.paired_dataset import load_image_pairs
from src.models.unet import unet_model


def train_unet(
    train_root="data/GOPRO_Large/train",
    max_images=50,
    batch_size=2,
    epochs=5,
    out_path="artifacts/unet.keras",
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    ds = load_image_pairs(train_root, max_images=max_images, normalize="0_1").batch(batch_size)

    model = unet_model()
    model.compile(optimizer="adam", loss="mse")
    model.fit(ds, epochs=epochs)

    model.save(out_path)
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    train_unet()
