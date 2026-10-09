import argparse
import torch

from models.srcnn import SRCNN
from models.espcn import ESPCN


def create_model(model_name):

    model_name = model_name.upper()

    if model_name == "SRCNN":
        return SRCNN()

    elif model_name == "ESPCN":
        return ESPCN()

    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model):

    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total_params


def main():

    parser = argparse.ArgumentParser(
        description="Count trainable parameters in a model"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["SRCNN", "ESPCN"],
        help="Model name"
    )

    args = parser.parse_args()

    model = create_model(args.model)

    params = count_parameters(model)

    print("=" * 40)
    print(f"Model: {args.model}")
    print(f"Trainable Parameters: {params:,}")
    print("=" * 40)


if __name__ == "__main__":
    main()