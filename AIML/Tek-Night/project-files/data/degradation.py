# High res image -> degraded low res image

from PIL import Image, ImageFilter
import numpy as np
import io
import random

def bicubic_downsample(image, scale_factor):
    """
    Downsample high res PIL image using bicubic interpolation

    Args:
        image(PIL.Image): high res input image
        scale_factor(int): downsampling factor

    Returns:
        PIL.Image: Downsampled low-res image
    """
    if scale_factor <=0:
        raise ValueError("scale_factor must be greater than 0.")
    
    width, height = image.size

    new_width = width //scale_factor
    new_height = height //scale_factor

    lr_image = image.resize(
        (new_width, new_height),
        Image.Resampling.BICUBIC
    )
    return lr_image


def apply_gaussian_blur(image, sigma):
    """
    Apply Gaussian blur to a PIL image.

    Args:
        image (PIL.Image): Input image
        sigma (float): Blur radius

    Returns:
        PIL.Image: blurred image
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    
    blurred_image = image.filter(
        ImageFilter.GaussianBlur(radius = sigma)
    )

    return blurred_image
    

def apply_jpeg_compression(image, quality):
    """
    Apply JPEG compression to a PIL image

    Args:
        image (PIL.Image): Input image
        quality(int): JPEG quality(1-100)

    Returns:
        PIL.Image: JPEG compressed image
    """

    if not(1 <= quality <= 100):
        raise ValueError("quality must be between 1 and 100")
    
    buffer = io.BytesIO() # fake file that exists in memory

    image.save(
        buffer,
        format = "JPEG",
        quality = quality,
    )

    buffer.seek(0)
    
    compressed = Image.open(buffer)

    return compressed.copy()


def apply_gaussian_noise(image, std):
    """
    Apply Gaussian noise to a PIL image.

    Args:
        image (PIL.Image): Input image.
        std (float): Standard deviation of the noise

    Returns:
        PIL.Image: Noise image.
    """

    if std < 0:
        raise ValueError("std must be non-negative")
    
    image_array = np.array(image).astype(np.float32)

    noise = np.random.normal(
        loc = 0,
        scale = std,
        size = image_array.shape
    )

    noisy = image_array + noise

    noisy = np.clip(noisy, 0, 255)

    noisy = noisy.astype(np.uint8)

    return Image.fromarray(noisy)


def degrade_image(image, config):
    """
    Apply the compleye degradation pipeline to an HR image.

    Args:
        image (PIL.Image): High-res image
        config (dict): Config directory

    Returns:
        PIL.Image: Degraded low-res image
    """

    degradation = config["degradation"]

    blur_cfg = degradation["blur"]
    jpeg_cfg = degradation["jpeg"]
    noise_cfg = degradation["noise"]

    # gaussian blur (optional)

    if blur_cfg["enabled"] and random.random() < blur_cfg["probability"]:
        image = apply_gaussian_blur(
            image,
            blur_cfg["sigma"]
        )
    
    # bicubic downsampling

    image = bicubic_downsample(
        image,
        degradation["scale_factor"]
    )

    # Optional JPEG compression
    
    if jpeg_cfg["enabled"] and random.random() < jpeg_cfg["probability"]:

        quality = random.randint(
            jpeg_cfg["quality_min"],
            jpeg_cfg["quality_max"]
        )

        image = apply_jpeg_compression(image = image,
                                       quality =  quality)

    # gaussian noise (optional)
    if noise_cfg["enabled"] and random.random() < noise_cfg["probability"]:
        image = apply_gaussian_noise(
            image,
            noise_cfg["std"]
        )

    return image
