import os
import json
import csv
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.srcnn import SRCNN
from models.espcn import ESPCN

from data.dataset import SuperResolutionDataset
from utils.metrics import calculate_psnr, calculate_ssim


os.makedirs("benchmarks", exist_ok=True)


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)



def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    elif torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")



def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )



def load_model(model, checkpoint_path, device):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model



def create_test_loader(config):

    test_dataset = SuperResolutionDataset(
        image_dir="dataset/test",
        config=config,
        train=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["training"].get(
            "num_workers",
            0
        )
    )

    return test_loader



def evaluate_bicubic(test_loader):

    total_psnr = 0.0
    total_ssim = 0.0


    with torch.no_grad():

        for lr, hr in test_loader:

            prediction = F.interpolate(
                lr,
                scale_factor=4,
                mode="bicubic",
                align_corners=False
            )


            for pred_img, target_img in zip(
                prediction.numpy(),
                hr.numpy()
            ):

                total_psnr += calculate_psnr(
                    pred_img,
                    target_img
                )

                total_ssim += calculate_ssim(
                    pred_img,
                    target_img
                )


    n = len(test_loader.dataset)


    return (
        total_psnr / n,
        total_ssim / n
    )



def evaluate(model, test_loader, device):

    criterion = nn.L1Loss()

    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0


    with torch.no_grad():

        for lr, hr in test_loader:

            lr = lr.to(device)
            hr = hr.to(device)


            # SRCNN needs bicubic upscaled input
            if isinstance(model, SRCNN):

                lr = F.interpolate(
                    lr,
                    scale_factor=4,
                    mode="bicubic",
                    align_corners=False
                )


            prediction = model(lr)


            loss = criterion(
                prediction,
                hr
            )

            total_loss += loss.item()


            prediction_np = (
                prediction
                .cpu()
                .numpy()
            )

            hr_np = (
                hr
                .cpu()
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


    n = len(test_loader.dataset)


    return (
        total_loss / len(test_loader),
        total_psnr / n,
        total_ssim / n
    )



def benchmark_speed(
        model,
        input_tensor,
        device,
        runs=100
):

    model.eval()

    input_tensor = input_tensor.to(device)


    with torch.no_grad():

        for _ in range(10):
            model(input_tensor)


    if device.type == "cuda":
        torch.cuda.synchronize()


    start = time.time()


    with torch.no_grad():

        for _ in range(runs):
            model(input_tensor)


    if device.type == "cuda":
        torch.cuda.synchronize()


    end = time.time()


    return (
        (end - start)
        / runs
        * 1000
    )



def main():

    device = get_device()

    print(
        f"Using device: {device}"
    )


    models = {

        "SRCNN": {
            "model": SRCNN(),
            "checkpoint": "checkpoints/srcnn_best.pth",
            "config": "configs/srcnn_x4.json",
            "input": torch.randn(1,3,128,128)
        },


        "ESPCN": {
            "model": ESPCN(),
            "checkpoint": "checkpoints/espcn_best.pth",
            "config": "configs/espcn_x4.json",
            "input": torch.randn(1,3,32,32)
        }

    }


    results = []


    # ==========================
    # Bicubic baseline
    # ==========================

    print("\n" + "="*40)
    print("BICUBIC")
    print("="*40)


    config = load_config(
        "configs/espcn_x4.json"
    )

    test_loader = create_test_loader(
        config
    )


    bicubic_psnr, bicubic_ssim = evaluate_bicubic(
        test_loader
    )


    print(
        f"Parameters : 0"
    )

    print(
        f"Test Loss  : N/A"
    )

    print(
        f"PSNR       : {bicubic_psnr:.4f}"
    )

    print(
        f"SSIM       : {bicubic_ssim:.4f}"
    )

    print(
        f"Speed      : N/A"
    )


    results.append(
        [
            "Bicubic",
            0,
            "N/A",
            bicubic_psnr,
            bicubic_ssim,
            "N/A"
        ]
    )



    # ==========================
    # SRCNN + ESPCN
    # ==========================

    for name, info in models.items():

        print("\n" + "="*40)
        print(name)
        print("="*40)


        config = load_config(
            info["config"]
        )


        model = load_model(
            info["model"],
            info["checkpoint"],
            device
        )


        params = count_parameters(
            model
        )


        test_loader = create_test_loader(
            config
        )


        loss, psnr, ssim = evaluate(
            model,
            test_loader,
            device
        )


        inference_time = benchmark_speed(
            model,
            info["input"],
            device
        )


        print(
            f"Parameters : {params:,}"
        )

        print(
            f"Test Loss  : {loss:.4f}"
        )

        print(
            f"PSNR       : {psnr:.4f}"
        )

        print(
            f"SSIM       : {ssim:.4f}"
        )

        print(
            f"Speed      : {inference_time:.4f} ms"
        )


        results.append(
            [
                name,
                params,
                loss,
                psnr,
                ssim,
                inference_time
            ]
        )



    with open(
        "benchmarks/results.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)


        writer.writerow(
            [
                "Model",
                "Parameters",
                "Test Loss",
                "PSNR",
                "SSIM",
                "Inference Time (ms)"
            ]
        )


        writer.writerows(
            results
        )


    print(
        "\nSaved results to benchmarks/results.csv"
    )



if __name__ == "__main__":
    main()