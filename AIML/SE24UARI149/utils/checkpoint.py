"""
checkpoint.py

Checkpoint save/load utilities. Designed with Colab in mind: free-tier
sessions can disconnect or time out mid-training, so checkpoints should be
written frequently to a path that survives the session (e.g. a mounted
Google Drive folder), and resuming should restore model, optimizer, and
epoch/step counters exactly, not just model weights.
"""

import os
from pathlib import Path

import torch


def save_checkpoint(path: str, model, optimizer, epoch: int, step: int,
                     best_val_psnr: float, model_cfg: dict, extra: dict = None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_val_psnr": best_val_psnr,
        "model_cfg": model_cfg,
    }
    if extra:
        state.update(extra)

    # Write to a temp file then atomically rename, so a Colab disconnect
    # mid-write can't leave a corrupted checkpoint file behind.
    tmp_path = f"{path}.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path: str, model, optimizer=None, map_location="cpu"):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return {
        "epoch": state.get("epoch", 0),
        "step": state.get("step", 0),
        "best_val_psnr": state.get("best_val_psnr", float("-inf")),
        "model_cfg": state.get("model_cfg", None),
    }


def checkpoint_exists(path: str) -> bool:
    return Path(path).is_file()
