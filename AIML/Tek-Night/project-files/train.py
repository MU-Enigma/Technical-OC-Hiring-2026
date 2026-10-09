import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import SuperResolutionDataset
from models.srcnn import SRCNN
from models.espcn import ESPCN
from utils.metrics import calculate_psnr, calculate_ssim
from utils.visualization import save_prediction_images


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_dataloaders(config):

    pin_memory = torch.cuda.is_available()

    train_dataset = SuperResolutionDataset(
        image_dir="dataset/train",
        config=config,
        train=True
    )

    val_dataset = SuperResolutionDataset(
        image_dir="dataset/val",
        config=config,
        train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=pin_memory
    )

    return train_loader, val_loader


def create_model(model_name, device):

    model_name = model_name.upper()

    if model_name == "SRCNN":
        model = SRCNN()

    elif model_name == "ESPCN":
        model = ESPCN()

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device)


def prepare_input(lr, model_name):
    """
    Apply model-specific preprocessing.
    """

    if model_name.upper() == "SRCNN":
        lr = F.interpolate(
            lr,
            scale_factor=4,
            mode="bicubic",
            align_corners=False
        )

    return lr


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    model_name
):

    model.train()

    running_loss = 0.0

    for lr, hr in train_loader:

        lr = lr.to(device)
        hr = hr.to(device)

        lr = prepare_input(lr, model_name)

        optimizer.zero_grad()

        prediction = model(lr)

        loss = criterion(prediction, hr)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def validate(
    model,
    val_loader,
    criterion,
    device,
    model_name
):

    model.eval()

    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0

    with torch.no_grad():

        for lr, hr in val_loader:

            lr = lr.to(device)
            hr = hr.to(device)

            lr = prepare_input(lr, model_name)

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
        running_loss / len(val_loader),
        running_psnr / len(val_loader.dataset),
        running_ssim / len(val_loader.dataset)
    )


def main():

    parser = argparse.ArgumentParser(
        description="Train a Super Resolution model"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration JSON file"
    )

    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)

    config = load_config(args.config)

    device = get_device()

    print(f"Using device: {device}")
    print(f"Training model: {config['model']['name']}")

    train_loader, val_loader = create_dataloaders(config)

    model = create_model(
        config["model"]["name"],
        device
    )

    criterion = nn.L1Loss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["optimizer"]["learning_rate"]
    )

    num_epochs = config["training"]["epochs"]

    best_val_loss = float("inf")

    checkpoint_path = os.path.join(
        "checkpoints",
        f"{config['model']['name'].lower()}_best.pth"
    )

    for epoch in range(num_epochs):

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            config["model"]["name"]
        )

        val_loss, psnr, ssim = validate(
            model,
            val_loader,
            criterion,
            device,
            config["model"]["name"]
        )

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"PSNR: {psnr:.2f} | "
            f"SSIM: {ssim:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )

            print(f"Best model saved to {checkpoint_path}")

        # Save sample prediction every 5 epochs
        if (epoch + 1) % 5 == 0:

            model.eval()

            with torch.no_grad():

                # Always use the first validation batch
                sample_lr, sample_hr = next(iter(val_loader))

                sample_lr = sample_lr.to(device)
                sample_hr = sample_hr.to(device)

                model_input = prepare_input(
                    sample_lr,
                    config["model"]["name"]
                )

                sample_prediction = model(model_input)

                save_prediction_images(
                    model_input[0],
                    sample_prediction[0],
                    sample_hr[0],
                    epoch + 1
                )


if __name__ == "__main__":
    main()