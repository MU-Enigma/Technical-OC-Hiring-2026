import argparse
import json
import os

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from models.srcnn import SRCNN
from models.espcn import ESPCN

from data.degradation import degrade_image
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


def create_model(model_name, device):

    if model_name.upper() == "SRCNN":
        model = SRCNN()

    elif model_name.upper() == "ESPCN":
        model = ESPCN()

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device)


def main():

    parser = argparse.ArgumentParser(
        description="Run inference on a single image."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to config file"
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained checkpoint"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image"
    )

    args = parser.parse_args()

    config = load_config(args.config)

    device = get_device()

    os.makedirs("outputs", exist_ok=True)

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

    model.eval()

    transform = transforms.ToTensor()

    hr_image = Image.open(args.image).convert("RGB")

    lr_image = degrade_image(
        hr_image,
        config
    )

    hr_tensor = transform(hr_image)

    lr_tensor = transform(lr_image)

    model_input = lr_tensor.unsqueeze(0).to(device)

    if config["model"]["name"].upper() == "SRCNN":

        model_input = F.interpolate(
            model_input,
            scale_factor=4,
            mode="bicubic",
            align_corners=False
        )

    with torch.no_grad():

        prediction = model(model_input)

    save_prediction_images(
        model_input[0].cpu(),
        prediction[0].cpu(),
        hr_tensor,
        epoch=0,
        output_dir="outputs"
    )

    print("Inference completed.")
    print("Results saved to outputs/")


if __name__ == "__main__":
    main()