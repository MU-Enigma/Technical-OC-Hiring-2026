"""
train_gan.py

Adversarial fine-tuning: takes an ALREADY pixel-loss-trained generator
checkpoint (e.g. runs/small_realistic/checkpoints/best.pt) and fine-tunes it
with a combined L1 + adversarial loss, using a PatchGAN discriminator
trained alongside it.

Warm-starting from a pretrained generator (rather than training adversarially
from a random initialization) directly follows SRGAN's own training recipe
(Ledig et al., 2017: pretrain with pixel loss, then fine-tune adversarially)
and is significantly more stable than adversarial training from scratch --
important given the limited hyperparameter-tuning budget available here.

Usage:
    python train_gan.py \
        --generator_checkpoint runs/small_realistic/checkpoints/best.pt \
        --degrade_config configs/degrade_realistic.json \
        --output_dir runs/small_gan \
        --epochs 20
"""

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
from sr_net import SRNet
from discriminator import PatchDiscriminator
from losses import GeneratorGANLoss, discriminator_loss
from metrics import compute_metrics
from checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists
from visualize import save_progression_grid


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_generator_checkpoint(path: str, device):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["model_cfg"]
    model = SRNet(num_blocks=cfg["num_blocks"], num_channels=cfg["num_channels"],
                  scale_factor=cfg["scale_factor"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    return model, cfg


def build_loaders(args, scale_factor):
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
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(generator, val_loader, device):
    generator.eval()
    total_psnr, total_ssim, n = 0.0, 0.0, 0
    for lr, hr in val_loader:
        lr, hr = lr.to(device), hr.to(device)
        pred = generator(lr)
        m = compute_metrics(pred, hr)
        total_psnr += m["psnr"]
        total_ssim += m["ssim"]
        n += 1
    generator.train()
    return {"psnr": total_psnr / max(1, n), "ssim": total_ssim / max(1, n)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_checkpoint", type=str, required=True,
                         help="Path to an already pixel-loss-trained generator "
                              "checkpoint to warm-start from")
    parser.add_argument("--degrade_config", type=str, required=True)
    parser.add_argument("--source_dir", type=str, default="source_images")
    parser.add_argument("--splits_dir", type=str, default="splits")
    parser.add_argument("--output_dir", type=str, default="runs/small_gan")
    parser.add_argument("--epochs", type=int, default=20,
                         help="Fine-tuning is typically shorter than the "
                              "initial pixel-loss training run")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_g", type=float, default=1e-5,
                         help="Generator LR during fine-tuning -- kept low "
                              "to avoid destroying the pretrained weights")
    parser.add_argument("--lr_d", type=float, default=1e-4)
    parser.add_argument("--w_l1", type=float, default=1.0)
    parser.add_argument("--w_adv", type=float, default=0.01,
                         help="Adversarial loss weight -- kept small so L1 "
                              "still dominates and anchors correctness")
    parser.add_argument("--hr_patch_size", type=int, default=128)
    parser.add_argument("--patches_per_image", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--progression_every", type=int, default=1)
    parser.add_argument("--checkpoint_every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "progression").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    with open(output_dir / "run_config.json", "w") as f:
        json.dump({"args": vars(args)}, f, indent=2)

    device = get_device()
    print(f"Using device: {device}")

    generator, gen_cfg = load_generator_checkpoint(args.generator_checkpoint, device)
    print(f"Warm-started generator from {args.generator_checkpoint} "
          f"({generator.count_parameters():,} params)")

    discriminator = PatchDiscriminator().to(device)
    print(f"Discriminator: {discriminator.count_parameters():,} params")

    train_loader, val_loader = build_loaders(args, gen_cfg["scale_factor"])
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d)
    gen_loss_fn = GeneratorGANLoss(w_l1=args.w_l1, w_adv=args.w_adv)

    start_epoch = 0
    best_val_psnr = float("-inf")
    latest_ckpt_path = output_dir / "checkpoints" / "latest.pt"
    if args.resume and checkpoint_exists(str(latest_ckpt_path)):
        state = load_checkpoint(str(latest_ckpt_path), generator, opt_g, map_location=device)
        start_epoch = state["epoch"] + 1
        best_val_psnr = state["best_val_psnr"]
        print(f"Resumed from epoch {state['epoch']}")

    log_path = output_dir / "logs" / "train_log.csv"
    if not log_path.exists() or not args.resume:
        with open(log_path, "w") as f:
            f.write("epoch,gen_l1,gen_adv,disc_loss,val_psnr,val_ssim,epoch_time_sec\n")

    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        generator.train()
        discriminator.train()
        running_gen_l1, running_gen_adv, running_disc = 0.0, 0.0, 0.0
        n_steps = 0

        for lr_batch, hr_batch in train_loader:
            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)

            # --- Discriminator step ---
            with torch.no_grad():
                fake_hr = generator(lr_batch)
            opt_d.zero_grad()
            real_logits = discriminator(hr_batch)
            fake_logits = discriminator(fake_hr.detach())
            d_loss = discriminator_loss(real_logits, fake_logits)
            d_loss.backward()
            opt_d.step()

            # --- Generator step ---
            opt_g.zero_grad()
            fake_hr = generator(lr_batch)
            fake_logits_for_g = discriminator(fake_hr)
            g_loss, g_components = gen_loss_fn(fake_hr, hr_batch, fake_logits_for_g)
            g_loss.backward()
            opt_g.step()

            running_gen_l1 += g_components["l1"]
            running_gen_adv += g_components["adversarial"]
            running_disc += d_loss.item()
            n_steps += 1
            global_step += 1

            if global_step % args.log_every == 0:
                print(f"  epoch {epoch} step {global_step} "
                      f"gen_l1={g_components['l1']:.4f} "
                      f"gen_adv={g_components['adversarial']:.4f} "
                      f"disc_loss={d_loss.item():.4f}")

        avg_gen_l1 = running_gen_l1 / max(1, n_steps)
        avg_gen_adv = running_gen_adv / max(1, n_steps)
        avg_disc = running_disc / max(1, n_steps)
        val_metrics = evaluate(generator, val_loader, device)
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch}: gen_l1={avg_gen_l1:.4f} gen_adv={avg_gen_adv:.4f} "
              f"disc_loss={avg_disc:.4f} val_psnr={val_metrics['psnr']:.2f} "
              f"val_ssim={val_metrics['ssim']:.4f} ({epoch_time:.1f}s)")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_gen_l1:.5f},{avg_gen_adv:.5f},{avg_disc:.5f},"
                    f"{val_metrics['psnr']:.4f},{val_metrics['ssim']:.5f},{epoch_time:.2f}\n")

        if (epoch + 1) % args.progression_every == 0:
            generator.eval()
            with torch.no_grad():
                sample_lr, sample_hr = next(iter(val_loader))
                sample_lr, sample_hr = sample_lr.to(device), sample_hr.to(device)
                sample_pred = generator(sample_lr)
            save_progression_grid(
                sample_lr, sample_pred, sample_hr,
                str(output_dir / "progression" / f"epoch_{epoch:03d}.png"),
                scale_factor=gen_cfg["scale_factor"],
            )
            generator.train()

        if (epoch + 1) % args.checkpoint_every == 0:
            is_best = val_metrics["psnr"] > best_val_psnr
            best_val_psnr = max(best_val_psnr, val_metrics["psnr"])
            save_checkpoint(str(latest_ckpt_path), generator, opt_g, epoch, global_step,
                             best_val_psnr, gen_cfg)
            if is_best:
                save_checkpoint(str(output_dir / "checkpoints" / "best.pt"), generator,
                                 opt_g, epoch, global_step, best_val_psnr, gen_cfg)

    print(f"GAN fine-tuning complete. Best val PSNR: {best_val_psnr:.2f}")
    print(f"Outputs written to: {output_dir}")
    print("NOTE: PSNR/SSIM are expected to be similar or slightly lower than "
          "the pixel-loss-only checkpoint -- adversarial fine-tuning trades "
          "some pixel fidelity for perceptual sharpness. Judge primarily by "
          "qualitative comparison, not these metrics alone.")


if __name__ == "__main__":
    main()