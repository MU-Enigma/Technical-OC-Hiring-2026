# Self-Supervised Image Super-Resolution

Single image super-resolution (4x) trained entirely from an ordinary folder of
photos — no premade LR/HR dataset is downloaded or used. Low-resolution training
inputs are synthesized on the fly from high-resolution source images using a
randomized degradation pipeline (blur → downsample → noise → JPEG compression).

Three models are trained and compared: a non-learned bicubic floor, an SRCNN
baseline (Dong et al.), and a final SRResNet model (Ledig et al. backbone +
ESPCN PixelShuffle upsampling + EDSR's no-batch-norm finding + VDSR's
residual/delta prediction). A controlled ablation additionally compares training
SRResNet on bicubic-only degradation vs. the full randomized pipeline, to
test whether realistic degradation actually improves generalization to real
low-quality photos.

Full reasoning behind every design decision — the degradation strategy,
architecture choices, loss function, and evaluation approach — is in
[`WRITEUP.md`](WRITEUP.md).

## Results

| Model                             | PSNR   | SSIM   |
|-----------------------------------|--------|--------|
| Bicubic (non-learned floor)       | 24.780 | 0.6456 |
| SRCNN (baseline)                  | 24.945 | 0.6528 |
| SRResNet (bicubic-only)           | 24.966 | 0.6467 |
| SRResNet (full degradation)       | 25.614 | 0.6750 |

![Training progression at epoch 100](assets/snapshots/epoch_200.png)

![Real-photo qualitative comparison](assets/real_photo_comparison.png)

### Qualitative single-image example

**1. Original HR photo (ground truth):**

![Original HR photo](assets/demo_hr.png)

**2. Degraded LR input (shown blown up for visibility — model actually sees it much smaller):**

![Degraded LR input](assets/demo_lr.png)

**3. Model's predicted HR reconstruction:**

![Model's predicted HR reconstruction](assets/demo_pred.png)

**4. Side-by-side comparison — original vs predicted:**

![Side-by-side comparison](assets/demo_sidebyside.png)

## Project structure

```
self-supervised-sr/
├── .gitignore
├── README.md
├── requirements.txt
├── WRITEUP.md
├── LICENSE
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   ├── perceptual.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── scripts/
│   ├── download_data.py
│   ├── train_srcnn.py
│   ├── train_srresnet_bicubic.py
│   ├── train_srresnet_full.py
│   └── run_eval.py
│
├── configs/
│   ├── bicubic.json
│   ├── srcnn.json
│   ├── srresnet_bicubic.json
│   └── srresnet_full.json
│
├── data/
│   ├── images/
│   │   └── .gitkeep
│   └── real_lowres/
│       └── .gitkeep
│
├── outputs/
│   └── .gitkeep
│
├── assets/
    ├── snapshots/
    │   └── .gitkeep
    ├── comparison_table.md
    └── real_photo_comparison.png

```

## Setup

```bash
git clone https://github.com/<you>/self-supervised-sr.git
cd self-supervised-sr
pip install -r requirements.txt
```

## Running it

**1. Download and prepare the HR image pool**
```bash
python scripts/download_data.py
```
This downloads Oxford-IIIT Pet, keeps only images at or above a minimum native
resolution (never upsamples a source image), caps the pool at 3,000 images, and
saves the filtered pool to `data/images/`.

**2. (Optional) Add real low-quality photos for qualitative evaluation**
Drop 5–10 genuinely low-quality images (old photos, screenshots, compressed
downloads) into `data/real_lowres/`. These are never used in training — used
only for the qualitative, no-ground-truth comparison at evaluation time.

**3. Train each model**
```bash
python -m src.train --config configs/bicubic.json
python scripts/train_srcnn.py
python scripts/train_srresnet_bicubic.py
python scripts/train_srresnet_full.py
```
Each run writes to `outputs/<run_name>/`: checkpoints, per-epoch snapshot
images, and a training history log.

**4. Evaluate**
```bash
python scripts/run_eval.py
```
Produces `outputs/eval/comparison_table.md` (PSNR/SSIM for all four models on
the held-out synthetic test split) and `outputs/eval/real_photo_comparison.png`
(all four models run side by side on the real low-quality photo set).

## Key design decisions (short version — full reasoning in WRITEUP.md)

- **No premade dataset.** LR images are synthesized from HR photos on the fly:
  blur → bicubic downsample (4x) → noise → JPEG re-compression, with every
  parameter randomized per sample.
- **Why not just bicubic downsampling?** A model trained on bicubic-only
  degradation overfits to "undo bicubic resizing" specifically and generalizes
  poorly to real low-quality images. Tested directly via a controlled ablation
  (`srresnet_bicubic.json` vs. `srresnet_full.json` — identical architecture
  and loss, differing only in degradation strength).
- **Data split at the file level**, before any patch cropping, using a
  deterministic filename hash — guarantees no patch-level leakage between
  train/val/test.
- **Architecture**: SRResNet combines a residual-block backbone, PixelShuffle
  upsampling (staged in sequential 2x steps), EDSR's no-batch-norm finding, and
  VDSR's residual/delta prediction — each idea attributed to one specific paper.
- **Loss**: L1 as the primary pixel loss (avoids the blurry "safe average"
  predictions MSE produces on this ill-posed problem), with a small VGG16
  perceptual loss term added on top (weight 0.01) to further counteract
  residual blurriness. See `WRITEUP.md` Section 2.2–2.3 for the full reasoning,
  including the full adversarial/GAN alternative that was considered and not used.

## Demo & Technical Walkthrough

- **Junior-level workshop presentation:** https://youtu.be/KBnrTUmo654
- **Technical walkthrough (approach, implementation, key design decisions, live inference demo):** https://youtu.be/zz1xbqdLvjo
