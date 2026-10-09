import numpy as np

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)


def calculate_psnr(prediction, target):
    """
    Calculate PSNR between two images.

    Args:
        prediction (numpy.ndarray): (C, H, W)
        target (numpy.ndarray): (C, H, W)

    Returns:
        float
    """

    prediction = np.transpose(prediction, (1, 2, 0))
    target = np.transpose(target, (1, 2, 0))

    return peak_signal_noise_ratio(
        target,
        prediction,
        data_range=1.0
    )


def calculate_ssim(prediction, target):
    """
    Calculate SSIM between two images.

    Args:
        prediction (numpy.ndarray): (C, H, W)
        target (numpy.ndarray): (C, H, W)

    Returns:
        float
    """

    prediction = np.transpose(prediction, (1, 2, 0))
    target = np.transpose(target, (1, 2, 0))

    return structural_similarity(
        target,
        prediction,
        channel_axis=-1,
        data_range=1.0,
    )