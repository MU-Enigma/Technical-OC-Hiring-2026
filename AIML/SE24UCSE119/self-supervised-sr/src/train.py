"""
train.py
--------
One generic training loop, driven entirely by a JSON config.

Usage:
    python -m src.train --config configs/srresnet_full.json
"""

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import list_images, split_files, SRDataset
from src.models import build_model
from src.utils import set_seed, get_device, psnr, ssim, save_comparison_grid


def call_model(model, model_name, lr_small, lr_bicubic):
    """Different architectures take different inputs - this is the single
    place that knows how to call each one."""
    if model_name in ("bicubic", "srcnn"):
        return model(lr_bicubic)
    else:  # srresnet / srresnet_lite
        return model(lr_small, lr_bicubic)


def build_optimizer(model, model_name, config):
    """
    SRCNN gets Dong et al.'s original differential learning rates: 1e-4 for
    the first two layers (patch extraction, non-linear mapping) and 1e-5 for
    the last (reconstruction) layer - the paper found the last layer needs a
    smaller rate for stable convergence. Every other model uses a single
    learning rate from the config.
    """
    if model_name == "srcnn":
        layers = list(model.net.children())
        conv_layers = [l for l in layers if isinstance(l, torch.nn.Conv2d)]
        assert len(conv_layers) == 3
        return torch.optim.Adam([
            {"params": conv_layers[0].parameters(), "lr": 1e-4},
            {"params": conv_layers[1].parameters(), "lr": 1e-4},
            {"params": conv_layers[2].parameters(), "lr": 1e-5},
        ])
    return torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-4))


def load_config(path):
    with open(path) as f:
        return json.load(f)


def train(config):
    set_seed(config.get("seed", 42))
    device = get_device()
    print(f"[train] device = {device}")
    print(f"[train] config = {json.dumps(config, indent=2)}")

    data_dir = config["data_dir"]
    files = list_images(data_dir)
    assert len(files) >= 20, (
        f"Only found {len(files)} images in {data_dir}. "
        f"Need at least ~300 for a real run (task requires 300-1000)."
    )
    train_files, val_files, test_files = split_files(
        files,
        val_frac=config.get("val_frac", 0.1),
        test_frac=config.get("test_frac", 0.1),
        seed=config.get("split_seed", 42),
    )
    print(f"[train] split: {len(train_files)} train / {len(val_files)} val / {len(test_files)} test")

    scale = config.get("scale", 4)
    crop = config.get("hr_crop", 128)
    strength = config.get("degradation_strength", "medium")

    train_ds = SRDataset(train_files, scale=scale, hr_crop=crop, strength=strength, train=True)
    val_ds = SRDataset(val_files, scale=scale, hr_crop=crop, strength=strength, train=False)

    train_loader = DataLoader(train_ds, batch_size=config.get("batch_size", 16),
                               shuffle=True, num_workers=config.get("num_workers", 2),
                               drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.get("batch_size", 16),
                             shuffle=False, num_workers=0)

    model_name = config["model"]
    model = build_model(model_name, scale=scale).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] model = {model_name}, trainable params = {n_params:,}")

    run_name = config.get("run_name", model_name)
    out_dir = os.path.join(config.get("output_dir", "outputs"), run_name)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    snap_dir = os.path.join(out_dir, "snapshots")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(snap_dir, exist_ok=True)

    if model_name == "bicubic":
        val_metrics = evaluate_loader(model, model_name, val_loader, device)
        print(f"[train] bicubic floor (no training) - val PSNR={val_metrics['psnr']:.3f} "
              f"SSIM={val_metrics['ssim']:.4f}")
        _save_log(out_dir, [{"epoch": 0, **val_metrics}])
        return

    optimizer = build_optimizer(model, model_name, config)
    loss_fn = torch.nn.L1Loss()

    epochs = config.get("epochs", 10)
    log_every = config.get("log_every_steps", 20)
    snapshot_every = config.get("snapshot_every_epochs", 1)

    history = []
    global_step = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for step, (lr_small, lr_bicubic, hr) in enumerate(train_loader):
            lr_small, lr_bicubic, hr = lr_small.to(device), lr_bicubic.to(device), hr.to(device)

            pred = call_model(model, model_name, lr_small, lr_bicubic)
            loss = loss_fn(pred, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            if (step + 1) % log_every == 0:
                elapsed = time.time() - t0
                print(f"[train] epoch {epoch}/{epochs} step {step+1}/{len(train_loader)} "
                      f"loss={loss.item():.4f} elapsed={elapsed:.0f}s")

        avg_train_loss = running_loss / len(train_loader)
        val_metrics = evaluate_loader(model, model_name, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            **val_metrics,
        })
        print(f"[train] === epoch {epoch} done - train_loss={avg_train_loss:.4f} "
              f"val_psnr={val_metrics['psnr']:.3f} val_ssim={val_metrics['ssim']:.4f} ===")

        if epoch % snapshot_every == 0:
            _save_snapshot(model, model_name, val_loader, device, snap_dir, epoch)

        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pt")
        torch.save({"model_state": model.state_dict(), "config": config, "epoch": epoch}, ckpt_path)
        torch.save({"model_state": model.state_dict(), "config": config, "epoch": epoch},
                   os.path.join(ckpt_dir, "latest.pt"))

    _save_log(out_dir, history)
    print(f"[train] finished in {time.time() - t0:.0f}s. outputs in {out_dir}")


def evaluate_loader(model, model_name, loader, device, max_batches=None):
    model.eval()
    psnr_sum, ssim_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for i, (lr_small, lr_bicubic, hr) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            lr_small, lr_bicubic, hr = lr_small.to(device), lr_bicubic.to(device), hr.to(device)
            pred = call_model(model, model_name, lr_small, lr_bicubic)
            pred = pred.clamp(0, 1)
            psnr_sum += psnr(pred, hr)
            ssim_sum += ssim(pred, hr)
            n += 1
    n = max(n, 1)
    return {"psnr": psnr_sum / n, "ssim": ssim_sum / n}


def _save_snapshot(model, model_name, loader, device, snap_dir, epoch):
    model.eval()
    with torch.no_grad():
        lr_small, lr_bicubic, hr = next(iter(loader))
        lr_small, lr_bicubic, hr = lr_small.to(device), lr_bicubic.to(device), hr.to(device)
        pred = call_model(model, model_name, lr_small, lr_bicubic).clamp(0, 1)
    path = os.path.join(snap_dir, f"epoch_{epoch:03d}.png")
    save_comparison_grid(lr_bicubic, pred, hr, path)
    print(f"[train] saved snapshot: {path}")


def _save_log(out_dir, history):
    path = os.path.join(out_dir, "history.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[train] saved training log: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to a JSON config file")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)