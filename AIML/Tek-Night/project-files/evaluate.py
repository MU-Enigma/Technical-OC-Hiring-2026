import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import SuperResolutionDataset
from models.srcnn import SRCNN
from models.espcn import ESPCN
from utils.metrics import calculate_psnr, calculate_ssim


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_test_loader(config):

    pin_memory = torch.cuda.is_available()

    test_dataset = SuperResolutionDataset(
        image_dir="dataset/test",
        config=config,
        train=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"].get("num_workers", 0),
        pin_memory=pin_memory
    )

    return test_loader


def create_model(model_name, device):

    model_name = model_name.upper()

    if model_name == "SRCNN":
        model = SRCNN()
    elif model_name == "ESPCN":
        model = ESPCN()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device)


def evaluate(
    model,
    test_loader,
    criterion,
    device,
    model_name
):

    model.eval()

    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0

    with torch.no_grad():

        for lr, hr in test_loader:

            lr = lr.to(device)
            hr = hr.to(device)

            if model_name.upper() == "SRCNN":

                lr = F.interpolate(
                    lr,
                    scale_factor=4,
                    mode="bicubic",
                    align_corners=False
                )

            prediction = model(lr)

            loss = criterion(prediction, hr)

            running_loss += loss.item()

            prediction_np = prediction.cpu().numpy()
            hr_np = hr.cpu().numpy()

            for pred_img, target_img in zip(prediction_np, hr_np):

                running_psnr += calculate_psnr(
                    pred_img,
                    target_img
                )

                running_ssim += calculate_ssim(
                    pred_img,
                    target_img
                )

    return (
        running_loss / len(test_loader),
        running_psnr / len(test_loader.dataset),
        running_ssim / len(test_loader.dataset)
    )


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate a Super Resolution model"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration JSON file"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )

    args = parser.parse_args()

    config = load_config(args.config)

    device = get_device()

    print(f"Using device: {device}")

    test_loader = create_test_loader(config)

    model = create_model(
        config["model"]["name"],
        device
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    criterion = nn.L1Loss()

    test_loss, psnr, ssim = evaluate(
        model,
        test_loader,
        criterion,
        device,
        config["model"]["name"]
    )

    print("=" * 50)
    print(f"Model      : {config['model']['name']}")
    print(f"Checkpoint : {args.checkpoint}")
    print("=" * 50)

    print(f"Test Loss : {test_loss:.4f}")
    print(f"PSNR      : {psnr:.2f} dB")
    print(f"SSIM      : {ssim:.4f}")


if __name__ == "__main__":
    main()