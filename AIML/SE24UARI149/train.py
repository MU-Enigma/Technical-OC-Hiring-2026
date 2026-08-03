

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent / "data"))
sys.path.insert(0, str(Path(__file__).parent / "models"))
sys.path.insert(0, str(Path(__file__).parent / "utils"))

from dataset import SRPatchDataset
from sr_net import build_model_from_config
from losses import build_loss_from_config
from metrics import compute_metrics
from checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists
from visualize import save_progression_grid


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_loaders(args, model_cfg):
    scale_factor = model_cfg["scale_factor"]
    splits_dir = Path(args.splits_dir)

    train_ds = SRPatchDataset(
        source_dir=args.source_dir,
        file_list_path=str(splits_dir / "train.txt"),
        degrade_config_path=args.degrade_config,
        scale_factor=scale_factor,
        hr_patch_size=args.hr_patch_size,
        patches_per_image=args.patches_per_image,
        split="train",
    )
    val_ds = SRPatchDataset(
        source_dir=args.source_dir,
        file_list_path=str(splits_dir / "val.txt"),
        degrade_config_path=args.degrade_config,
        scale_factor=scale_factor,
        hr_patch_size=args.hr_patch_size,
        patches_per_image=max(1, args.patches_per_image // 4),
        split="val",
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, val_loader, device, loss_fn):
    model.eval()
    total_loss, total_psnr, total_ssim, n_batches = 0.0, 0.0, 0.0, 0
    for lr, hr in val_loader:
        lr, hr = lr.to(device), hr.to(device)
        pred = model(lr)
        loss, _components = loss_fn(pred, hr)
        m = compute_metrics(pred, hr)
        total_loss += loss.item()
        total_psnr += m["psnr"]
        total_ssim += m["ssim"]
        n_batches += 1
    model.train()
    return {
        "loss": total_loss / max(1, n_batches),
        "psnr": total_psnr / max(1, n_batches),
        "ssim": total_ssim / max(1, n_batches),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--degrade_config", type=str, required=True)
    parser.add_argument("--source_dir", type=str, default="source_images")
    parser.add_argument("--splits_dir", type=str, default="splits")
    parser.add_argument("--output_dir", type=str, default=None,
                         help="Defaults to runs/<model_name>_<degrade_name>")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hr_patch_size", type=int, default=128)
    parser.add_argument("--patches_per_image", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_every", type=int, default=50,
                         help="Steps between console loss logs")
    parser.add_argument("--progression_every", type=int, default=1,
                         help="Epochs between saving progression image grids")
    parser.add_argument("--checkpoint_every", type=int, default=1,
                         help="Epochs between checkpoint saves")
    parser.add_argument("--resume", action="store_true",
                         help="Resume from latest.pt in output_dir if present")
    args = parser.parse_args()

    with open(args.model_config) as f:
        model_cfg_raw = json.load(f)
    with open(args.degrade_config) as f:
        degrade_cfg_raw = json.load(f)

    if args.output_dir is None:
        run_name = f"{model_cfg_raw['name']}_{degrade_cfg_raw['name']}"
        args.output_dir = f"runs/{run_name}"

    output_dir = Path(args.output_dir)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "progression").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    # Persist the exact configs used for this run alongside its outputs,
    # so results are traceable back to the settings that produced them.
    with open(output_dir / "run_config.json", "w") as f:
        json.dump({
            "model_config": model_cfg_raw,
            "degrade_config": degrade_cfg_raw,
            "args": vars(args),
        }, f, indent=2)

    device = get_device()
    print(f"Using device: {device}")

    model, model_cfg = build_model_from_config(args.model_config)
    model.to(device)
    print(f"Model: {model_cfg['name']} ({model.count_parameters():,} parameters)")

    train_loader, val_loader = build_loaders(args, model_cfg)
    print(f"Train samples: {len(train_loader.dataset)}, "
          f"Val samples: {len(val_loader.dataset)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = build_loss_from_config(model_cfg_raw)
    print(f"Loss function: {type(loss_fn).__name__} "
          f"(loss_weights={model_cfg_raw.get('loss_weights', 'none -> pure L1')})")

    start_epoch = 0
    best_val_psnr = float("-inf")
    latest_ckpt_path = output_dir / "checkpoints" / "latest.pt"
    if args.resume and checkpoint_exists(str(latest_ckpt_path)):
        state = load_checkpoint(str(latest_ckpt_path), model, optimizer,
                                 map_location=device)
        start_epoch = state["epoch"] + 1
        best_val_psnr = state["best_val_psnr"]
        print(f"Resumed from epoch {state['epoch']}, best_val_psnr={best_val_psnr:.3f}")

    log_path = output_dir / "logs" / "train_log.csv"
    if not log_path.exists() or not args.resume:
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss,val_psnr,val_ssim,epoch_time_sec\n")

    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        n_steps = 0

        for lr_batch, hr_batch in train_loader:
            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)

            optimizer.zero_grad()
            pred = model(lr_batch)
            loss, components = loss_fn(pred, hr_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_steps += 1
            global_step += 1

            if global_step % args.log_every == 0:
                comp_str = " ".join(f"{k}={v:.4f}" for k, v in components.items())
                print(f"  epoch {epoch} step {global_step} {comp_str}")

        train_loss = running_loss / max(1, n_steps)
        val_metrics = evaluate(model, val_loader, device, loss_fn)
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_psnr={val_metrics['psnr']:.2f} "
            f"val_ssim={val_metrics['ssim']:.4f} "
            f"({epoch_time:.1f}s)"
        )

        with open(log_path, "a") as f:
            f.write(
                f"{epoch},{train_loss:.5f},{val_metrics['loss']:.5f},"
                f"{val_metrics['psnr']:.4f},{val_metrics['ssim']:.5f},"
                f"{epoch_time:.2f}\n"
            )

        if (epoch + 1) % args.progression_every == 0:
            model.eval()
            with torch.no_grad():
                sample_lr, sample_hr = next(iter(val_loader))
                sample_lr, sample_hr = sample_lr.to(device), sample_hr.to(device)
                sample_pred = model(sample_lr)
            save_progression_grid(
                sample_lr, sample_pred, sample_hr,
                str(output_dir / "progression" / f"epoch_{epoch:03d}.png"),
                scale_factor=model_cfg["scale_factor"],
            )
            model.train()

        if (epoch + 1) % args.checkpoint_every == 0:
            is_best = val_metrics["psnr"] > best_val_psnr
            best_val_psnr = max(best_val_psnr, val_metrics["psnr"])
            save_checkpoint(
                str(latest_ckpt_path), model, optimizer, epoch, global_step,
                best_val_psnr, model_cfg,
            )
            if is_best:
                save_checkpoint(
                    str(output_dir / "checkpoints" / "best.pt"),
                    model, optimizer, epoch, global_step, best_val_psnr, model_cfg,
                )

    print(f"Training complete. Best val PSNR: {best_val_psnr:.2f}")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()