"""
data.py
--------
Everything related to turning a folder of ordinary (high-res) images into
paired (low-res, high-res) training samples for super-resolution.

Key design decisions (see WRITEUP.md for full justification):

1. We NEVER download or use a pre-made LR/HR dataset. LR images are
   synthesized on-the-fly from HR images using a randomized degradation
   pipeline that mimics real-world image capture + compression.

2. Degradation pipeline = blur -> bicubic downsample -> noise -> JPEG
   compression, with RANDOMIZED parameters per sample. Using only clean
   bicubic downsampling (no blur/noise/compression) makes the task too easy
   and the model overfits to "undo bicubic downsampling" specifically,
   which generalizes poorly to real low-quality images (real camera/phone
   photos, old scans, compressed web images) that were never produced by
   pure bicubic downsampling. Randomizing degradation strength acts as an
   implicit data-augmentation over "how images get degraded", improving
   robustness.

   The three strength presets map directly onto our ablation and our
   reference papers:
     - "clean"  = the degradation assumption behind SRCNN/VDSR/ESPCN's
                  original bicubic-only training setup. Zero blur, zero
                  noise, zero compression - used for the bicubic-only
                  ablation arm (see WRITEUP.md, Section 4.3).
     - "medium" = our default training setup, following the
                  BSRGAN/Real-ESRGAN argument that realistic degradation
                  (blur + noise + JPEG, randomized) generalizes better to
                  real-world low-quality images.
     - "heavy"  = a harder setting, available for extra robustness testing,
                  not used in the core submission runs.

3. Splitting happens at the FILE level, before any patch extraction, so
   that no two patches from the same source image can end up in different
   splits (a common and easy-to-miss leakage bug in SR/patch-based tasks).
"""

import hashlib
import io
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torch

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder):
    folder = Path(folder)
    return sorted(
        [p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS]
    )


def split_files(files, val_frac=0.1, test_frac=0.1, seed=42):
    """
    Deterministic, file-level split using a hash of the filename rather than
    a random shuffle. This means:
      - The split is 100% reproducible without saving a random state.
      - Adding new images later doesn't reshuffle existing assignments
        (important for avoiding accidental leakage if the dataset grows).
    No image (and therefore none of its future patches/crops) can appear
    in more than one split.
    """
    def bucket(path):
        h = hashlib.md5((str(seed) + path.name).encode()).hexdigest()
        return int(h, 16) % 1000 / 1000.0

    train, val, test = [], [], []
    for f in files:
        b = bucket(f)
        if b < test_frac:
            test.append(f)
        elif b < test_frac + val_frac:
            val.append(f)
        else:
            train.append(f)
    return train, val, test


def degrade(hr_img: Image.Image, scale: int, rng: random.Random,
            strength="medium"):
    """
    Turn a high-res PIL image into a low-res PIL image.

    Pipeline: Gaussian blur -> bicubic downsample by `scale` -> additive
    Gaussian noise -> JPEG re-compression. Parameters are randomized within
    a range controlled by `strength` so that the model sees a *distribution*
    of degradations rather than one fixed deterministic transform.
    """
    w, h = hr_img.size

    if strength == "clean":
        # Zero blur, zero noise, zero compression - the exact bicubic-only
        # assumption SRCNN/VDSR/ESPCN trained under.
        blur_sigma = (0.0, 0.0)
        noise_std = (0.0, 0.0)
        jpeg_q = (100, 100)
    elif strength == "heavy":
        blur_sigma = (0.5, 2.5)
        noise_std = (2, 12)
        jpeg_q = (30, 70)
    else:  # medium (default)
        blur_sigma = (0.2, 1.4)
        noise_std = (0, 6)
        jpeg_q = (55, 95)

    img = hr_img

    # 1. Blur (simulates lens/motion blur, sensor point-spread function)
    sigma = rng.uniform(*blur_sigma)
    if sigma > 0.05:
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))

    # 2. Downsample (bicubic emulates a real camera/optical downsampling
    #    reasonably well; nearest-neighbor would create an unrealistically
    #    easy task - blocky artifacts are trivial to reverse and don't
    #    resemble real low-res images)
    lr_w, lr_h = max(1, w // scale), max(1, h // scale)
    img = img.resize((lr_w, lr_h), Image.BICUBIC)

    # 3. Sensor / transmission noise
    std = rng.uniform(*noise_std)
    if std > 0.1:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, std, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # 4. JPEG re-compression (only fires if q < 100)
    q = rng.randint(*jpeg_q)
    if q < 100:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

    return img


class SRDataset(Dataset):
    """
    On-the-fly LR/HR pair generator.

    Each __getitem__ call:
      1. loads an HR source image
      2. takes a random HR crop of size `hr_crop` (so the model sees varied
         content and we can train at a fixed tensor size for batching)
      3. degrades it to produce the LR input
      4. returns (lr_small_tensor, lr_bicubic_tensor, hr_tensor)

    For validation/test we disable random cropping/augmentation and use a
    fixed center crop for reproducible evaluation.
    """

    def __init__(self, files, scale=4, hr_crop=128, strength="medium",
                 train=True, seed=0):
        self.files = files
        self.scale = scale
        self.hr_crop = hr_crop
        self.strength = strength
        self.train = train
        self.seed = seed

    def __len__(self):
        return len(self.files)

    def _load(self, path):
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        path = self.files[idx]
        img = self._load(path)
        w, h = img.size
        crop = self.hr_crop

        # pad tiny images up so a crop is always possible
        if w < crop or h < crop:
            scale_up = max(crop / w, crop / h) + 0.01
            img = img.resize((int(w * scale_up) + 1, int(h * scale_up) + 1),
                              Image.BICUBIC)
            w, h = img.size

        if self.train:
            x0 = random.randint(0, w - crop)
            y0 = random.randint(0, h - crop)
            rng = random.Random()  # fresh randomness per call for degradation
        else:
            x0 = (w - crop) // 2
            y0 = (h - crop) // 2
            rng = random.Random(self.seed + idx)  # deterministic for val/test

        hr = img.crop((x0, y0, x0 + crop, y0 + crop))

        if self.train and random.random() < 0.5:
            hr = TF.hflip(hr)

        lr = degrade(hr, self.scale, rng, strength=self.strength)

        # THREE tensors returned, because our two learned models want the
        # LR information at different resolutions:
        #   - SRCNN is a "pre-upsampling" architecture (Dong et al.): it
        #     expects the input already bicubic-upsampled to full size,
        #     then refines it with a few conv layers.
        #   - SRResNet-lite is a "post-upsampling" architecture (Ledig et
        #     al. / Shi et al.): it operates on the small LR image directly
        #     and uses PixelShuffle to upsample only at the end, adding a
        #     residual/delta on top of a bicubic upsample computed
        #     separately.
        lr_small_tensor = TF.to_tensor(lr)                                   # (3, crop/scale, crop/scale)
        lr_bicubic_up = lr.resize((crop, crop), Image.BICUBIC)
        lr_bicubic_tensor = TF.to_tensor(lr_bicubic_up)                      # (3, crop, crop)
        hr_tensor = TF.to_tensor(hr)                                         # (3, crop, crop)
        return lr_small_tensor, lr_bicubic_tensor, hr_tensor

    def get_raw_pair(self, idx):
        """Returns (small LR PIL image, HR PIL image), useful for visualization."""
        path = self.files[idx]
        img = self._load(path)
        w, h = img.size
        crop = self.hr_crop
        if w < crop or h < crop:
            scale_up = max(crop / w, crop / h) + 0.01
            img = img.resize((int(w * scale_up) + 1, int(h * scale_up) + 1),
                              Image.BICUBIC)
            w, h = img.size
        x0, y0 = (w - crop) // 2, (h - crop) // 2
        hr = img.crop((x0, y0, x0 + crop, y0 + crop))
        rng = random.Random(self.seed + idx)
        lr = degrade(hr, self.scale, rng, strength=self.strength)
        return lr, hr