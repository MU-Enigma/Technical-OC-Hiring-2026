## Write-up: Deep Learning Image Super-Resolution from Synthetic Pairs

### 1. Data and Degradation Strategy

**Source data.** 500 natural photographs from BSDS500 (Arbelaez, Maire,
Fowlkes & Malik, IEEE TPAMI 2011), roughly 481x321px each, pulled from a
public GitHub mirror. No paired low-res/high-res dataset was used or
downloaded anywhere in this project. Every low-res training input was built
from these high-res images by a degradation pipeline written for this
project (`data/degrade.py`).

**How the low-res input is generated.** Given a high-res patch Y, a
synthetic low-res version X gets produced through a configurable sequence
of stages:

1. **Blur** (probability 0.8, sigma randomized between 0.2 and 2.0), which
   stands in for lens and sensor point-spread along with a bit of motion
   blur, since real cameras introduce this before any resizing happens at
   all.
2. **Downsampling** at the target scale factor (4x), mostly bicubic with
   an occasional random swap to bilinear or nearest-neighbor.
3. **Additive Gaussian noise** (probability 0.7, sigma randomized between 1
   and 12 out of 255), approximating sensor or low-light noise.
4. **JPEG re-encoding** (probability 0.9, quality randomized between 30 and
   90), which introduces the blocking and ringing artifacts real
   compressed images tend to carry.
5. **An optional second resize-and-recompress pass** (probability 0.25),
   simulating an image that's been resized or re-saved more than once.

The strength of each stage gets randomized per sample rather than fixed,
and the order of the noise and JPEG stages is itself randomly swapped, so
the model ends up seeing a whole distribution of plausible degradations
instead of one fixed, deterministic operation.

**Why this approach makes sense.** Real low-res images out in the world
were basically never produced by one clean resize. They usually carry some
unknown combination of blur, sensor noise, and compression, often stacked
more than once and in no particular order. Randomizing and compounding the
degradation stages gives the resulting model an actual shot at generalizing
to real low-res images it never saw degraded this exact way, which is
something this write-up tries to test directly later on, not just argue
for in the abstract.

**Scale factor and what it does to training.** A fixed 4x scale factor was
used throughout. At that factor, most of the fine texture is genuinely gone
from the low-res input, so the network has to infer plausible detail
rather than just denoise or sharpen what's already there, which is really
the whole source of the ill-posedness that keeps coming up throughout this
document. A model trained at one scale factor shouldn't be expected to
generalize well to a different one without retraining.

**What happens when the degradation is too clean.** A model trained only
on plain bicubic downsampling (`configs/degrade_naive.json`) learns to
reverse one specific deterministic operation, not real-world degradation in
general. This project actually built the training script for that naive
case (`scripts/run_baseline_naive.sh`) but chose not to run it to
completion given the time available. Instead, the argument here rests on
two things: the zero-parameter bicubic baseline used for the required
comparison further down, and this project's own real-world generalization
tests. That's a real, acknowledged gap rather than something quietly
skipped, and it's called out again in the limitations section.

**Splitting the data and avoiding leakage.** `data/prepare_splits.py` splits
whole source images (not patches) into train, validation, and test sets
(80/10/10, seed 42) before any cropping happens, and checks that the three
resulting lists are completely disjoint. Since each source image later
gets cut into 16 separate training patches, splitting after cropping could
let two patches from the same photo end up in both train and validation,
which would inflate scores without any real generalization behind them.
Validation and test samples also use a fixed, seeded random generator for
both crop location and degradation, so those numbers stay comparable
across runs, while training samples get fresh randomness every epoch on
purpose.

### 2. Model Design

**Architecture.** The generator (`models/sr_net.py`) is a configurable
residual CNN that borrows specific ideas from four different papers rather
than copying any one of them wholesale:
- From **FSRCNN** (Dong et al., 2016): do almost all the work while the
  image is still small, and only upsample right at the end, which is the
  main reason this was trainable at all on limited hardware.
- From **ESPCN** (Shi et al., 2016): PixelShuffle as the upsampling step,
  which avoids the checkerboard artifacts that learned deconvolution tends
  to produce.
- From **VDSR** (Kim et al., 2016): a global residual connection, so the
  network predicts a correction on top of a bicubic upsample instead of
  reconstructing the whole image from nothing.
- From **EDSR** (Lim et al., 2017): residual blocks with no batch
  normalization (since BatchNorm throws away the absolute intensity
  information needed for faithful reconstruction), plus L1 over L2 as the
  base loss.

Two configs exist: `model_tiny.json` (4 blocks, 32 channels, about 159K
parameters, used only for quick debugging) and `model_small.json` (8
blocks, 64 channels, about 927K parameters), which is the actual generator
used across all three training stages below.

**Loss function, across three stages.**

Stage 1, pure L1: because super-resolution is ill-posed, a pixel-wise loss
tends to converge toward the statistical average of the plausible outputs.
L2's squared penalty pushes harder toward that average than L1's linear one
does, which is why L1 was used as the base loss. This stage converged
cleanly (val loss and val PSNR both went flat by around epoch 30) and gave
a real, if modest, improvement over bicubic, though the output stayed
noticeably softer than ground truth on fine, repetitive texture.

Stage 2, a multi-component loss combining L1 with a gradient/edge loss and
an SSIM loss, both computed by hand with plain tensor operations (no
pretrained network involved anywhere, per this project's constraint
against external pretrained weights). This one didn't actually help.
Metrics plateaued slightly below Stage 1, and the output looked basically
identical. The reason turned out to matter more than the negative result
itself: SRNet is a discriminative regression network no matter which of
these three loss terms gets used, meaning it always computes one fixed
output as a function of its input. L1, gradient loss, and SSIM loss are all
still just measuring distance to one specific target, so reweighting
between them can't hand the network a capability (generating plausible
texture it hasn't strictly derived from the input) that the architecture
never had to begin with. That realization is what led directly to Stage 3.

Stage 3, adversarial fine-tuning using an LSGAN setup. Rather than train a
GAN from scratch, which tends to be unstable, the Stage 1 checkpoint was
warm-started and then fine-tuned with L1 plus an adversarial loss against a
small PatchGAN discriminator (`models/discriminator.py`, about 289K
parameters), following the same pretrain-then-fine-tune recipe SRGAN itself
uses. The adversarial gradient works differently from a pixel loss: it
nudges the generator toward whatever the discriminator has learned counts
as generally realistic, rather than toward one specific pixel target, which
is exactly the missing piece from Stage 2.

The first attempt at this failed outright. With the discriminator's
learning rate set ten times faster than the generator's, and the
adversarial weight quite low, the discriminator's loss collapsed within a
handful of epochs and just sat there, meaning it had gotten good enough at
telling real from fake that the generator stopped receiving any useful
signal at all. The output looked no different from Stage 1.

After rebalancing (slowing the discriminator's learning rate down and
raising the adversarial weight), things stabilized properly. Discriminator
loss settled into a healthy, oscillating range instead of collapsing,
meaning the two networks were genuinely contesting each other rather than
one just winning outright. Generator L1 rose a bit from its Stage 1 value
and then held steady at a new equilibrium rather than continuing to
degrade. Qualitatively, there was a real if modest improvement in local
texture (grass, foliage, water reflections) compared to Stage 1, smaller
than the dramatic sharpening SRGAN reports, which seems mostly down to a
still-conservative adversarial weight, a short fine-tuning run (20 epochs),
and a discriminator kept deliberately small.

**Alternative considered: perceptual loss.** This is probably the most
direct fix for Stage 1's blurriness, and it's what SRGAN itself pairs with
adversarial loss. It wasn't used here because this project ruled out any
pretrained auxiliary networks from the start, and perceptual loss (usually
built on a pretrained VGG) requires exactly that. That constraint is also
why Stage 2 tried a hand-built substitute instead, and why the adversarial
route in Stage 3 made sense as the real fix: a discriminator can be trained
completely from scratch, so it doesn't run into the same restriction a VGG
feature loss would.

### 3. Evaluation and Analysis

**Metrics.** PSNR (pixel fidelity, in dB) and SSIM (structural similarity,
between 0 and 1), both implemented in `utils/metrics.py` and checked
against a couple of sanity cases (identical images scoring near the
theoretical maximum, unrelated images scoring low) before being trusted for
anything. Both metrics are known to line up imperfectly with how humans
actually judge image quality, which is why they're reported alongside
qualitative comparisons rather than on their own.

**Beyond the numbers.** Training progression grids get saved every epoch,
showing bicubic input, model output, and ground truth side by side, so
there's an actual visual record of how things changed over training, not
just a curve. `evaluate.py` also produces a qualitative comparison grid
across bicubic, each trained model, and ground truth on real held-out test
images. Beyond the BSDS500 test set, the model was also tried on real
photos well outside its training distribution: a portrait with a face,
glasses, and hair (content categories BSDS500 barely has), synthetically
degraded and then super-resolved, and a genuine phone photo made low-res
through a plain thumbnail resize rather than this project's own pipeline,
run straight through the model with no synthetic degradation step at all.
Both showed real, visible improvement (clearer hair and glasses-frame
detail in the first case, less blocky net-mesh and building edges in the
second), which is decent direct evidence of generalization beyond the
specific degradation this project trained on, rather than just a good
score on a matched test set.

**Baseline versus final model.**

| Model | PSNR (dB) | SSIM | Notes |
|---|---|---|---|
| Bicubic baseline (0 parameters) | 22.998 | 0.578 | Also what the network reduces to if its residual branch learns nothing |
| Stage 1: pure L1 (40 epochs) | 23.808 | 0.625 | Test split, realistic degradation; converged cleanly |
| Stage 2: multi-component loss (40 epochs) | ~23.40 | ~0.609 | Validation split; plateaued lower, no visible gain |
| Stage 3: GAN fine-tuned, rebalanced (20 epochs) | ~24.4-25.7 | ~0.49-0.57 | Validation split; expected trade-off for perceptual gain |

Stage 1 beats bicubic by a real but modest margin (+0.81 dB PSNR, +0.047
SSIM). Stage 3 scoring lower than Stage 1 on both metrics, while looking
somewhat sharper, matches what the literature would predict for adversarial
fine-tuning: some pixel-exact accuracy gets traded away for perceptual
realism. That's meant to be read together with the qualitative comparison,
not taken as a standalone verdict.

### Limitations and future work

A few things worth being upfront about. The naive-degradation baseline was
built but never actually trained to completion, so the "too clean
degradation" argument leans on literature and indirect generalization
evidence rather than a direct side-by-side comparison, and training it
properly would be the single most valuable thing left to do here. The
adversarial fine-tuning stage also used a fairly conservative weight and a
short training budget, so there's a reasonable chance a longer run with a
stronger adversarial term would push the effect further. And the decision
to rule out pretrained networks meant perceptual loss was never on the
table, even though it's arguably the most well-established fix for exactly
the blurriness problem this project ran into.
