import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def tensor_to_image(tensor):
    """
    Convert PyTorch tensor (C,H,W) to numpy image (H,W,C)
    """

    image = tensor.detach().cpu()

    image = image.permute(1, 2, 0).numpy()

    image = image.clip(0, 1)

    return image


def bicubic_upscale(lr_tensor, scale_factor=4):
    """
    Apply bicubic interpolation baseline.

    Args:
        lr_tensor: Tensor (C,H,W)
        scale_factor: Upscaling factor

    Returns:
        numpy image (H,W,C)
    """

    image = tensor_to_image(lr_tensor)

    image = (image * 255).astype(np.uint8)

    pil_image = Image.fromarray(image)

    upscaled = pil_image.resize(
        (
            image.shape[1] * scale_factor,
            image.shape[0] * scale_factor
        ),
        Image.Resampling.BICUBIC
    )

    upscaled = np.array(upscaled)

    upscaled = upscaled.astype(np.float32) / 255.0

    return upscaled


def save_prediction_images(
    lr,
    prediction,
    hr,
    epoch,
    output_dir="outputs"
):
    """
    Save LR, Bicubic, Prediction and Ground Truth comparison.
    """

    os.makedirs(output_dir, exist_ok=True)

    lr_img = tensor_to_image(lr)

    bicubic_img = bicubic_upscale(
        lr,
        scale_factor=4
    )

    prediction_img = tensor_to_image(prediction)

    hr_img = tensor_to_image(hr)


    plt.figure(figsize=(16, 4))


    # Low Resolution
    plt.subplot(1, 4, 1)
    plt.imshow(
        lr_img,
        interpolation="nearest"
    )
    plt.title("LR (32×32)")
    plt.axis("off")


    # Bicubic baseline
    plt.subplot(1, 4, 2)
    plt.imshow(bicubic_img)
    plt.title("Bicubic")
    plt.axis("off")


    # Model output
    plt.subplot(1, 4, 3)
    plt.imshow(prediction_img)
    plt.title("Model Output")
    plt.axis("off")


    # Ground truth
    plt.subplot(1, 4, 4)
    plt.imshow(hr_img)
    plt.title("Ground Truth")
    plt.axis("off")


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            output_dir,
            f"epoch_{epoch:03d}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()