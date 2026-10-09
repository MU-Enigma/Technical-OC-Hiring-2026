import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import SuperResolutionDataset
from utils.metrics import calculate_psnr, calculate_ssim


def load_config(path):

    with open(path, "r") as f:
        return json.load(f)


def bicubic_upscale(lr):

    return F.interpolate(
        lr,
        scale_factor=4,
        mode="bicubic",
        align_corners=False
    )


def main():

    config = load_config(
        "configs/espcn_x4.json"
    )


    dataset = SuperResolutionDataset(
        image_dir="dataset/test",
        config=config,
        train=False
    )


    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )


    total_psnr = 0
    total_ssim = 0


    with torch.no_grad():

        for lr, hr in loader:


            prediction = bicubic_upscale(
                lr
            )


            prediction_np = (
                prediction
                .numpy()
            )

            hr_np = (
                hr
                .numpy()
            )


            for pred_img, target_img in zip(
                prediction_np,
                hr_np
            ):

                total_psnr += calculate_psnr(
                    pred_img,
                    target_img
                )

                total_ssim += calculate_ssim(
                    pred_img,
                    target_img
                )


    n = len(dataset)


    print("="*40)
    print("Bicubic Baseline")
    print("="*40)

    print(
        f"PSNR : {total_psnr/n:.4f} dB"
    )

    print(
        f"SSIM : {total_ssim/n:.4f}"
    )


if __name__ == "__main__":
    main()