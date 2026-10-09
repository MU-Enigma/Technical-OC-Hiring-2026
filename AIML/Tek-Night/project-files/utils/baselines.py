from PIL import Image
import numpy as np


def bicubic_upscale(lr_tensor, scale_factor=4):

    """
    Upscale LR tensor using bicubic interpolation.

    Input:
        lr_tensor: torch tensor (C,H,W)

    Output:
        numpy image (H,W,C)
    """

    image = lr_tensor.detach().cpu()

    image = image.permute(1,2,0).numpy()

    image = (image * 255).astype(np.uint8)

    pil_image = Image.fromarray(image)

    hr_image = pil_image.resize(
        (
            image.shape[1] * scale_factor,
            image.shape[0] * scale_factor
        ),
        Image.Resampling.BICUBIC
    )

    hr_image = np.array(hr_image)

    hr_image = hr_image.astype(np.float32) / 255.0

    return hr_image