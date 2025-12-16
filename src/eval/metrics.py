#!/usr/bin/env python
# coding: utf-8

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def to_0_1(x: np.ndarray) -> np.ndarray:
    """Convert [-1,1] -> [0,1] if needed; if already [0,1], keep."""
    x = np.asarray(x)
    if x.min() < 0:
        x = (x + 1.0) / 2.0
    return np.clip(x, 0.0, 1.0)


def compute_psnr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = to_0_1(y_true)
    y_pred = to_0_1(y_pred)
    return psnr(y_true, y_pred, data_range=1.0)


def compute_ssim(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = to_0_1(y_true)
    y_pred = to_0_1(y_pred)
    return ssim(y_true, y_pred, channel_axis=-1, data_range=1.0)
