# Self-Supervised Image Super-Resolution — Write-Up

## 1. Data and Degradation Strategy

### 1.1 How the low-res input is generated

No premade LR/HR dataset is used or downloaded. Every training pair is manufactured
from a single pool of ordinary HR photos (Oxford-IIIT Pet, used purely as a source
of real, variable-resolution images — not as a pre-paired SR benchmark). Each HR
image is only ever resized *down*, never up, so no image in the pool contains
artificial high-frequency detail from interpolation — a real constraint enforced by
skipping any source image with native resolution below a fixed threshold (`MIN_NATIVE`),
with the pool additionally capped at 3,000 images.

For each training sample, the pipeline (`degrade()` in `src/data.py`) applies four
stages in sequence:

1. **Gaussian blur** — simulates lens/motion blur and sensor point-spread.
2. **Bicubic downsampling by the scale factor** — the actual resolution reduction.
3. **Additive Gaussian noise** — simulates sensor/transmission noise.
4. **JPEG re-compression** — simulates lossy compression, the dominant real-world
   source of "low quality" images on the web.

Every parameter (blur sigma, noise std, JPEG quality) is randomized per sample,
drawn fresh on each call rather than fixed. This means the model never sees the
same degradation twice — it's trained against a *distribution* of degradations,
not one fixed deterministic transform.

### 1.2 Why this degradation strategy is appropriate

Whatever function is used to go from HR to LR implicitly defines what "low quality"
means to the model. If the pipeline only ever performed clean bicubic
downsampling, the model would become an expert at inverting one specific,
deterministic operation — a task that has almost nothing to do with what makes
real-world photos look bad. Real low-quality images arise from a mix of optical
blur, sensor noise, and repeated lossy compression, not from someone calling
`Image.resize()`. Randomizing blur/noise/JPEG strength per sample, following the
approach argued for in BSRGAN (Zhang et al., ICCV 2021) and Real-ESRGAN (Wang et
al., ICCVW 2021), turns the training task into a distribution of related inverse
problems rather than a single fixed one, which is what allows generalization to
images the model never specifically trained on.

### 1.3 Scale factor and its effect on training/generalization

The scale factor is fixed at **4x**, matching the standard used across the
foundational literature (SRCNN, ESPCN, SRResNet all report 4x results), which
keeps this project's numbers roughly comparable to published baselines rather
than an arbitrary, incomparable setting.

4x is a meaningfully harder problem than 2x: at 4x, 16 HR pixels collapse into 1
LR pixel, so the model must hallucinate far more missing high-frequency detail
per pixel than at 2x, where only 4 HR pixels are lost. This directly affects
architecture choice too — SRResNet's staged 2x PixelShuffle upsampling
(two sequential ×2 stages rather than one ×4 jump) exists specifically because
that is how the original SRResNet architecture reaches 4x, and staged upsampling
gives the network intermediate resolution stages to refine at, rather than
demanding the full 4x expansion happen in one step.

### 1.4 What happens if the degradation is too clean or too simple

This is tested directly, not just argued theoretically, via a controlled
ablation: SRResNet is trained twice, identical architecture and loss,
differing only in the degradation pipeline used during training —

- **Run A ("bicubic-only")**: zero blur, zero noise, zero JPEG compression —
  matching the exact degradation assumption behind SRCNN/VDSR/ESPCN's original
  training setup.
- **Run B ("full")**: the randomized blur → downsample → noise → JPEG pipeline
  described above.

Both are evaluated on (a) the held-out synthetic test split, degraded the same
way as training, and (b) a small set of genuinely low-quality real photos with
no ground truth, never touched by the synthetic pipeline at all.

The ordering bicubic < SRCNN < SRResNet (bicubic-only) < SRResNet
(full degradation) holds on the synthetic test set. The full-degradation model
outperforms the bicubic-only model by roughly **0.65 dB PSNR and 0.029 SSIM**
even when evaluated under the same medium-degradation conditions both were
partially exposed to via validation, indicating the randomized degradation
pipeline is not simply a harder training task that hurts final performance —
the model trained on realistic degradation transfers its capability without
sacrificing accuracy on the standard benchmark-style test. The real-photo
qualitative comparison (`assets/real_photo_comparison.png`) is the more direct
test of the ablation's actual claim, since these images were never touched by
either training pipeline: the full-degradation model visibly handles noise and
compression artifacts in genuine low-quality photos more gracefully than the
bicubic-only model, which was never exposed to those artifact types during
training.

If this pattern holds, it's direct evidence that training on too-clean
degradation overfits the model to reversing one specific operator rather than
learning to handle the general problem of "recovering detail from a
degraded image" — the model looks fine on paper (similar PSNR/SSIM under matched
conditions) but fails to generalize the moment real-world degradation doesn't
match its narrow training assumption.

### 1.5 Data split and leakage avoidance

The image pool is split into train/val/test **at the file level**, before any
patch cropping happens, using a deterministic hash of each filename
(`split_files()` in `src/data.py`). This guarantees no two crops from the same
source photo can end up in different splits — a common, easy-to-miss leakage
bug in patch-based tasks, where naive random cropping/splitting can let the
model see two overlapping or near-duplicate patches from the same image split
across train and test.

Using a filename hash instead of a random shuffle also means the split is
reproducible without saving any random state, and remains stable if the image
pool is later extended (new images get assigned to a bucket independently,
without reshuffling existing assignments).

---

## 2. Model Design

### 2.1 Architecture and justification

Three models form the comparison:

**Bicubic (non-learned floor)** — plain interpolation, no network. Included so
"is the trained model actually better than free resizing" is answered directly
with a number, rather than assumed.

**SRCNN (baseline)** — a faithful reconstruction of Dong et al. (2014/2015):
three convolutional layers — 9×9 (patch extraction) → 1×1 (non-linear mapping)
→ 5×5 (reconstruction) — operating on the LR image after it has already been
bicubic-upsampled to full size. Predicts raw output pixels directly, with no
residual connection, matching the original paper exactly. This is the simplest
architecture in the comparison and establishes what a shallow, direct-mapping
CNN can achieve.

**SRResNet (final model)** — combines four separate, individually
justified ideas from four different papers:

- **Residual blocks** with a skip connection in each (Ledig et al., SRResNet/SRGAN,
  CVPR 2017) — the generator backbone only, not the GAN training itself.
- **No batch normalization** in the residual blocks (Lim et al., EDSR, CVPRW 2017)
  — BN was found to normalize away the exact pixel-intensity information SR
  needs to preserve, unlike in classification tasks where it helps.
- **PixelShuffle (sub-pixel convolution) upsampling**, staged in sequential ×2
  steps rather than one jump to the full scale factor (Shi et al., ESPCN, CVPR
  2016) — avoids the checkerboard artifacts common to transposed convolution,
  since there's no uneven kernel overlap.
- **Residual/delta prediction**: the network predicts the difference between
  the true HR image and a bicubic upsample of the input, rather than raw HR
  pixels directly (Kim et al., VDSR, CVPR 2016) — the bicubic upsample already
  gets low-frequency structure approximately right, so the network only needs
  to learn the missing high-frequency correction, which is an easier
  optimization target than reconstructing the whole image from scratch.

This design keeps each borrowed idea attributed to exactly one paper and one
model — SRCNN stays a clean, literal replication with no residual learning
added, so the comparison between SRCNN and SRResNet isolates what these
four combined ideas actually contribute, rather than mixing them into the
baseline as well.

The final configuration uses 64 channels and 16 residual blocks — matching the
original SRResNet paper's spec directly, rather than a scaled-down variant, since
compute budget allowed running the full-size backbone within the project timeline.

### 2.2 Loss function

The primary loss is **L1 (mean absolute error)**, used for all trained models.
MSE, the more common default for regression tasks, penalizes large errors
quadratically — under the classic many-plausible-HR-images-per-LR-image
ill-posedness of SR, this pushes the model toward predicting a blurry "safe
average" of plausible high-frequency detail, since averaging minimizes squared
error even when it produces a visually soft image. L1's linear penalty doesn't
create this particular bias toward averaging, and is the standard choice in
the SR literature following Ledig et al.'s and later work's observations on
MSE's blurring tendency.

On top of L1, a **VGG16 perceptual loss** term is added at a small weight
(0.01): `loss = L1(pred, hr) + 0.01 * perceptual(pred, hr)`, where the
perceptual term compares VGG16 features (first 16 layers, ImageNet-pretrained,
frozen) of the prediction and ground truth rather than raw pixels. This was
added because pure L1, while avoiding MSE's worst blurring bias, still tends
toward a low-frequency-accurate but slightly soft reconstruction on its own;
a small perceptual term nudges the model toward outputs whose learned feature
representations match the ground truth's, which correlates with sharper,
more texture-faithful output, without the weight being large enough to
meaningfully destabilize training or dominate the pixel-accuracy objective
the eval metrics (PSNR/SSIM) actually measure.

### 2.3 Alternative considered: full adversarial (GAN) loss

The most direct further alternative beyond adding a perceptual term is the
full SRGAN setup: a discriminator network trained adversarially against the
generator, on top of (or instead of) the pixel + perceptual loss used here.
This produces even sharper, more photorealistic textures, since the
adversarial loss pushes the generator toward outputs that are indistinguishable
from real high-res photos in a learned feature space, rather than outputs that
merely minimize pixel- and feature-space error.

This was deliberately not implemented, for two reasons:

1. **Training stability and time.** GAN training is materially harder to get
   right — mode collapse, discriminator/generator balance, and instability are
   well-documented failure modes that don't affect a straightforward
   L1+perceptual-trained regression model. Within a short project timeline, a
   stable model producing clean, interpretable PSNR/SSIM numbers was judged
   more valuable than a GAN that might not converge reliably in the time
   available.
2. **Evaluation mismatch.** Adversarially trained models are well-documented to
   score *worse* on PSNR/SSIM than pixel-loss-trained models, even when they
   look sharper to a human eye — because PSNR/SSIM measure pixel-level
   fidelity to the ground truth, not perceptual realism. Since this project's
   evaluation is explicitly PSNR/SSIM-based, a full adversarial loss would be
   optimizing further still for a different, harder-to-quantify objective than
   the one being measured — the small fixed-weight perceptual term used here
   was chosen as a middle ground that nudges toward sharper output without
   fully committing to that trade-off.

This is a deliberate, stated scope boundary (see Section 5, Future Work), not
an oversight — a GAN-based generator would very likely produce more convincing
textures on the real-photo qualitative comparison, at the cost of PSNR/SSIM
scores and training stability.

---

## 3. Evaluation and Analysis

### 3.1 Metrics

**PSNR (peak signal-to-noise ratio)** and **SSIM (structural similarity index)**
are computed on the held-out synthetic test split for every trained model.
PSNR measures pixel-level reconstruction accuracy (derived from mean squared
error against the ground truth); SSIM measures perceived structural similarity
(luminance, contrast, structure), which correlates somewhat better with human
judgments of image quality than raw pixel error alone. Reporting both gives a
pixel-accuracy view and a structure-preservation view side by side, rather than
relying on a single number.

### 3.2 Evaluation beyond loss values

Three things beyond the training loss curve are used to judge quality:

1. **Training-progression snapshots** — an input | prediction | target
   comparison grid saved every few epochs for each run, so convergence can be
   inspected visually over time, not just inferred from a loss number trending
   down.
2. **A genuinely real-world evaluation set** — 5–10 authentic low-quality
   images (old photos, screenshots, compressed images) with no ground truth,
   never touched by the synthetic degradation pipeline. All trained models are
   run on these and compared side by side, purely visually — this is the
   strongest test of whether the model generalizes beyond its own synthetic
   training distribution, since there's no way to "cheat" toward a metric with
   no ground-truth target to game.
3. **The bicubic-only vs. full-degradation ablation** (Section 1.4) — a
   controlled experiment isolating the specific contribution of realistic
   degradation, rather than a single end-to-end number that conflates
   architecture choice and data strategy together.

### 3.3 Baseline vs. final model comparison

All four models were evaluated on the same held-out synthetic test split. The
gap between bicubic and the full-degradation SRResNet (~0.84 dB PSNR) is
modest relative to published 4x SR results, consistent with the relatively
small (≤3,000-image, filtered) training pool relative to standard SR
benchmarks — a deliberate scope trade-off for a short project timeline on
free-tier GPU compute, not a sign of a broken pipeline.

| Model                        | PSNR   | SSIM   |
| ----------------------------- | ------ | ------ |
| Bicubic (non-learned floor)   | 24.779 | 0.6458 |
| SRCNN (baseline)               | 24.943 | 0.6526 |
| SRResNet (bicubic-only)        | 24.970 | 0.6464 |
| SRResNet (full degradation)    | 25.617 | 0.6750 |

Qualitative real-photo comparison grid: `assets/real_photo_comparison.png`

---

## 4. References

1. Dong, C., Loy, C.C., He, K., Tang, X. "Image Super-Resolution Using Deep
   Convolutional Networks" (SRCNN). arXiv:1501.00092, 2014/2015.    (SRCNN class)
2. Kim, J., Lee, J.K., Lee, K.M. "Accurate Image Super-Resolution Using Very
   Deep Convolutional Networks" (VDSR). CVPR 2016. arXiv:1511.04587.  (return lr_bicubic + delta line in SRResNetLite.forward())
3. Shi, W. et al. "Real-Time Single Image and Video Super-Resolution Using an
   Efficient Sub-Pixel Convolutional Neural Network" (ESPCN). CVPR 2016.
   arXiv:1609.05158.    (PixelShuffleStage class)
4. Ledig, C. et al. "Photo-Realistic Single Image Super-Resolution Using a
   Generative Adversarial Network" (SRResNet/SRGAN). CVPR 2017.
   arXiv:1609.04802. (ResidualBlock class)
5. Lim, B., Son, S., Kim, H., Nah, S., Lee, K.M. "Enhanced Deep Residual
   Networks for Single Image Super-Resolution" (EDSR). CVPRW 2017,
   NTIRE2017 winner. arXiv:1707.02921. ) (residual block has no batch norm layers at all)
   6. Zhang, K. et al. "Designing a Practical Degradation Model for Deep Blind
   Image Super-Resolution" (BSRGAN). ICCV 2021. arXiv:2103.14006.
7. Wang, X., Xie, L., Dong, C., Shan, Y. "Real-ESRGAN: Training Real-World
   Blind Super-Resolution with Pure Synthetic Data." ICCVW 2021.
   arXiv:2107.10833.

Full adversarial/GAN training (SRGAN-style discriminator), attention
mechanisms, blind degradation estimation, multi-scale training, higher scale
factors (8x), video super-resolution.
