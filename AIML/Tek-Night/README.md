# Image upscaling using deep learning

## Overview

Computer vision task that reconstructs high-res images from low-res images.

Objective: Upscale images by a 4x factor while preserving important details such as edges, textures and structures

Approaches used and compared:

- Bicubic interpolation
- SRCNN
- ESPCN

Evaluated both Reconstruction quality and computational efficiency

---

# Problem statement

Low-res images lose information due to downsampling and compression

The models implemented aim to recover a high res version of the low res images.


# Dataset

## DIV2K dataset
    Used DIV2K dataset for 800 HR images.
   
Split: 
   Training: 700
   Validation: 50
   Testing: 50

## How to download? 

Paste this in your browser: **https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip**

## Preprocessing

High res image
   |
   |
128x128 HR patch
   |
   |
Downsampling x4
   |
   |
32x32 LR image


The task is then formulated as:

LR image -> neural network -> HR image

# Models

## Bicubic interpolation

Estimates missing pixels using surrounding pixel information without learning from data.

Used as a baseline for comparing neural network based super-resolution methods.

## SRCNN

Super resolution convolutional neural network is one of the classic deep learning models for image super-resolution.

- Operates on high resolution space after bicubic upscaling
- Uses convolution layers to learn high-frequency details
- Uses residual learning by predicting additional details over the bicubic image

Architecture:

Input LR image (32x32x3)

↓

Bicubic upscaling

↓

Conv(9x9)
ReLU

↓

Conv(5x5)
ReLU

↓

Conv(3x3)
ReLU

↓

Conv(5x5)

↓

Residual addition

↓

Output HR image (128x128x3)

Trainable parameters: 138,947


## ESPCN

Efficient sub-pixel convolutional neural network improves efficiency by performing feature extraction in low resolution space and applying upscaling at the end.

- Lower computational cost
- Efficient upscaling using PixelShuffle
- Uses residual learning with bicubic interpolation as the base image

Architecture:

Input LR image (32x32x3)

↓

Feature extraction using convolution layers

↓

PixelShuffle x4

↓

Residual addition

↓

Output HR image (128x128x3)

Trainable parameters: 180,208

# Training 

## Framework

- PyTorch

- GPU: RTX 4070 Laptop

## Hyperparameters

Loss function: L1 Loss 

   Directly minimizes absolute pixel-wise difference.
   Chosen over L2 loss because L1 Loss is less sensitive to outliers and sometimes gives sharper reconstructions.

Optimizer: Adam

Lr: 0.0001

Epochs: 50

Batch size: 16


# Evaluation metrics

## PSNR

- Peak signal-to-noise ratio measures pixel level similarity 
- Higher is better

## SSIM

- Structural similarity index measures preservation of luminance, contrat, structural info.
- Higher is better

## Inference time

Avg forward pass latency

Lower is better

# Result analysis

Quantitative results:

| Model | Parameters | PSNR | SSIM | Inference Time |
|---|---:|---:|---:|---:|
| Bicubic | 0 | 23.9695 | 0.6266 | N/A |
| SRCNN | 138,947 | 24.8741 | 0.6688 | 0.8351 ms |
| ESPCN | 180,208 | 24.3840 | 0.6483 | 0.6551 ms |


Observations:

- Bicubic interpolation provides a strong baseline because PSNR and SSIM are pixel-based metrics.
- SRCNN achieved the highest reconstruction quality with the best PSNR and SSIM scores.
- ESPCN achieved faster inference compared to SRCNN because it performs feature extraction in low-resolution space and uses PixelShuffle for upscaling.
- ESPCN provides a better efficiency-quality tradeoff while SRCNN focuses more on reconstruction quality.

Neural networks can achieve further improvements with larger datasets, deeper architectures and more advanced degradation pipelines.

# Installation

```bash
git clone <repository-url>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

# Training

For ESPCN:

```bash
python train.py --config configs/espcn_x4.json
```

For SRCNN:

```bash
python train.py --config configs/srcnn_x4.json
```

# Evaluation

```bash
python evaluate.py \
--config configs/espcn_x4.json \
--checkpoint checkpoints/espcn_best.pth
```

# Inference

For ESPCN:

```bash
python inference.py --config configs\espcn_x4.json --checkpoint checkpoints\espcn_best.pth --image demo_images\your_image.png
```

For SRCNN:

```bash
python inference.py --config configs\srcnn_x4.json --checkpoint checkpoints\srcnn_best.pth --image demo_images\your_image.png
```
# Benchmark

```bash
python benchmark.py
```
Make sure to be in Tek-night/project-files for the commands to work
Outputs are saved in **outputs/epoch_000.png** for now


# What can be improved:

- Training on larger datasets
- Deeper architectures can be implemented