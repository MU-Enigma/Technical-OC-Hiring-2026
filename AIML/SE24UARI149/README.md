## Image Super-Resolution from Synthetic Pairs

This project builds a 4x super-resolution pipeline trained entirely on
synthetically-degraded image pairs. No premade low-res/high-res dataset was
available or allowed, so given just a folder of ordinary photographs, the
project constructs its own (low-res, high-res) training pairs using a
configurable degradation pipeline, trains a small residual CNN to reverse
that degradation, and then works through three successive training setups:
plain pixel loss, a hand-built multi-component loss, and adversarial (GAN)
fine-tuning. Results from all three are reported honestly, including a
setup that didn't actually help and a GAN training run that failed on the
first attempt before being fixed.

Check `WRITEUP.md` for the full technical write-up (data strategy, model
design, the loss-function iteration story, evaluation) and
`SR_Project_Deep_Dive_Report.pdf` for a much longer, fully-cited walkthrough
that goes layer by layer and explains every metric and loss term
mathematically.


Here is the video link of the explanation of the project : https://youtu.be/tqlPPZkMS5I
Here is the link to the workshop style presentation of the project : https://docs.google.com/presentation/d/13o-hKf88N8rZ6ZAyySdAq4-wxA7-yayO/edit?usp=sharing&ouid=106820110769257223817&rtpof=true&sd=true

### Project structure

```
sr-project/
├── configs/
│   ├── degrade_naive.json          # bicubic-only degradation (baseline / strawman)
│   ├── degrade_realistic.json      # randomized, compounded degradation (main pipeline)
│   ├── model_tiny.json             # 4 blocks / 32 channels, fast dev config
│   ├── model_small.json            # 8 blocks / 64 channels, main reported model (pure L1)
│   └── model_small_multiloss.json  # same architecture, L1 + gradient + SSIM loss
├── data/
│   ├── prepare_splits.py           # leakage-safe train/val/test split by source image
│   ├── degrade.py                  # degradation pipeline (both configs)
│   └── dataset.py                  # PyTorch Dataset: patch crop + degrade + augment
├── models/
│   ├── sr_net.py                   # generator: residual CNN + PixelShuffle upsampler
│   ├── discriminator.py            # PatchGAN discriminator (used only for adversarial fine-tuning)
│   └── losses.py                   # L1, combined (gradient+SSIM), and LSGAN adversarial losses
├── utils/
│   ├── metrics.py                  # PSNR / SSIM
│   ├── checkpoint.py               # save/load/resume, safe against a dropped Colab session
│   └── visualize.py                # progression / comparison image grids
├── scripts/
│   ├── download_sample_data.sh     # reproduces the BSDS500 source_images/ folder used here
│   ├── run_baseline_naive.sh       # small model + naive degradation (built but not run to completion, see WRITEUP)
│   ├── run_final_realistic.sh      # small model + realistic degradation, pure L1 (this is "V1")
│   ├── run_multiloss.sh            # V2: L1 + gradient + SSIM loss
│   ├── run_gan_finetune.sh         # V3: adversarial fine-tuning, rebalanced hyperparameters
│   ├── run_tiny_dev.sh             # fast tiny-model debug run
│   └── run_evaluation.sh           # baseline / model comparison
├── notebooks/
│   ├── colab_train.ipynb           # thin Colab wrapper (mounts Drive, calls into the .py files)
│   └── presentation.ipynb          # live demo: metrics tables plus an interactive "try any image" cell
├── train.py                        # training loop for V1 and V2
├── train_gan.py                    # adversarial fine-tuning loop for V3 (warm-starts from V1)
├── evaluate.py                     # baseline vs. trained model(s) comparison
├── infer.py                        # single-image demo: degrade synthetically, then upscale, compare to ground truth
├── infer_real.py                   # single-image demo: a genuinely low-res photo, no ground truth available
├── requirements.txt
├── README.md
└── WRITEUP.md
```

### Setup

```bash
pip install -r requirements.txt
```

Needs Python 3.9+. Torch will pick up a GPU automatically if one's
available (say, on a Colab T4 runtime) and otherwise just falls back to
CPU, no code changes needed either way.

### Step-by-step: running the full pipeline

#### 1. Get some source images

Either run `bash scripts/download_sample_data.sh`, which pulls down the
exact 500-image BSDS500 folder this project was built and tested against
(Arbelaez, Maire, Fowlkes & Malik, IEEE TPAMI 2011), or just drop your own
photos into `source_images/` (jpg/png/bmp, at least 128px on each side).

#### 2. Build the train/val/test split

```bash
python3 data/prepare_splits.py --source_dir source_images --out_dir splits --train_frac 0.8 --val_frac 0.1 --seed 42
```

This splits whole source images, not patches, into `splits/train.txt`,
`splits/val.txt`, and `splits/test.txt`, and asserts all three lists are
disjoint from each other. That ordering matters quite a bit: see
`WRITEUP.md` for why splitting after cropping patches would leak
information between the splits.

#### 3. Quick smoke test (optional but recommended)

```bash
bash scripts/run_tiny_dev.sh
```

Good for catching a broken data pipeline early, before committing to a
longer run.

#### 4. Train V1, pure L1 loss

```bash
bash scripts/run_final_realistic.sh
```

This is the main reported baseline model.

#### 5. Train V2, the multi-component loss

```bash
bash scripts/run_multiloss.sh
```

Worth saying upfront: this one didn't actually improve on V1. It's included
anyway because figuring out why it didn't help turned out to be the most
useful finding of the whole project (full explanation in `WRITEUP.md`).

#### 6. Train V3, adversarial fine-tuning

```bash
bash scripts/run_gan_finetune.sh
```

Needs `runs/small_realistic/checkpoints/best.pt` to already exist, since it
warm-starts from V1 rather than training a GAN from scratch. Uses the
hyperparameters that actually worked after a first attempt failed (the
comment block at the top of the script tells that story, and so does
`WRITEUP.md` in more depth).

#### 7. Evaluate everything against a common test set

```bash
bash scripts/run_evaluation.sh
```

or call it directly with whichever checkpoints you want to compare:

```bash
python3 evaluate.py --source_dir source_images --splits_dir splits \
  --test_degrade_config configs/degrade_realistic.json \
  --checkpoints runs/small_realistic/checkpoints/best.pt runs/small_gan/checkpoints/best.pt \
  --labels pure_l1 gan_finetuned --output_dir eval_results_comparison
```

#### 8. Try it on one image

```bash
# High-res source, synthetically degraded, with ground truth shown alongside:
python3 infer.py --checkpoint runs/small_realistic/checkpoints/best.pt \
  --image source_images/108073.jpg --degrade_config configs/degrade_realistic.json \
  --output_dir demo_output

# A genuinely low-res photo, run as-is, no ground truth to compare against:
python3 infer_real.py --checkpoint runs/small_realistic/checkpoints/best.pt \
  --image your_real_low_res_photo.jpg --output_dir demo_output_real
```

#### 9. Live demo for a presentation

Open `notebooks/presentation.ipynb`. It loads whatever checkpoints you've
trained, shows the metrics comparison, and has a cell where you can drop in
any image path and see it super-resolved on the spot (it figures out on its
own whether to synthetically degrade a high-res image first or just run a
low-res one directly, based on a 500px size cutoff).

### Running on Colab

1. Push the repo to GitHub (private is fine), or copy it straight into
   Google Drive.
2. Open `notebooks/colab_train.ipynb`, switch the runtime to a T4 GPU.
3. Run the cells in order: mount Drive, get the code onto the VM, install
   requirements, then run through the `scripts/*.sh` entry points one by
   one, finishing with evaluation.

Checkpoints get written under the Drive-mounted path where relevant, so a
dropped session doesn't cost you a finished run. Just add `--resume` to
pick a training script back up where it left off.

### A few notes on reproducibility

Splitting and validation/test degradation both use fixed seeds, so those
numbers stay stable and comparable across runs. Training patches, on the
other hand, get fresh randomness every epoch on purpose, since that variety
is exactly what makes the realistic degradation pipeline useful. Every run
also writes its exact configuration out to `runs/<name>/run_config.json`,
so you can always trace a result back to what actually produced it.

One more thing worth flagging: `scripts/run_baseline_naive.sh` exists and
runs correctly, but was never trained to completion for the results
reported here. That was a deliberate time trade-off, not an oversight, and
`WRITEUP.md` explains the reasoning behind leaning on the bicubic baseline
instead.