"""
losses.py

Multi-component loss combining:
  - L1 (pixel-wise fidelity, anchors output to correct absolute values)
  - Gradient/edge loss (penalizes the model for producing smoother edges
    than the ground truth -- directly targets the blur-averaging effect of
    pure pixel-wise loss under super-resolution's ill-posedness)
  - SSIM loss (structural/contrast similarity, reuses utils/metrics.py's
    differentiable SSIM implementation as a loss term rather than just an
    eval metric)

No pretrained/external networks are used anywhere in this file -- every
term is computed directly from the model's own output and the ground
truth, via plain tensor arithmetic (finite differences for the gradient
term) or the hand-written SSIM already built and verified in this project.

This directly targets the plateau documented in WRITEUP.md: a pure-L1
model converged (flat val_loss/val_psnr across epochs 30-39) at a modest
improvement over bicubic, which is the expected signature of L1's bias
toward the "safe blurry average" of plausible high-frequency solutions
under an ill-posed inverse problem, not undertraining.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from metrics import ssim as differentiable_ssim


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 distance between horizontal/vertical finite-difference gradients
    of pred and target. Penalizes the model specifically for smoothing out
    edges relative to the ground truth, rather than pixel value in general.
    Pure tensor arithmetic -- no learned parameters, no external network.
    """
    def grad(x):
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy

    pred_dx, pred_dy = grad(pred)
    target_dx, target_dy = grad(target)
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


class CombinedLoss(nn.Module):
    """Weighted sum of L1 + gradient loss + (1 - SSIM).

    Default weights keep L1 as the dominant term (anchoring correctness)
    with gradient/SSIM terms as smaller corrective pressure toward sharper,
    more structurally faithful output -- large weights on the secondary
    terms can destabilize training, so start small and increase only if the
    loss curves stay stable.
    """

    def __init__(self, w_l1: float = 1.0, w_gradient: float = 0.5, w_ssim: float = 0.2):
        super().__init__()
        self.w_l1 = w_l1
        self.w_gradient = w_gradient
        self.w_ssim = w_ssim

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        l1 = F.l1_loss(pred, target)
        grad = gradient_loss(pred, target)
        # differentiable_ssim returns per-sample similarity in [0,1] (or up
        # to ~1); (1 - ssim) turned into a loss term, mean over the batch.
        ssim_val = differentiable_ssim(
            torch.clamp(pred, 0.0, 1.0), torch.clamp(target, 0.0, 1.0)
        ).mean()
        ssim_loss = 1.0 - ssim_val

        total = self.w_l1 * l1 + self.w_gradient * grad + self.w_ssim * ssim_loss

        components = {
            "l1": l1.item(),
            "gradient": grad.item(),
            "ssim_loss": ssim_loss.item(),
            "total": total.item(),
        }
        return total, components


def build_loss_from_config(cfg: dict) -> nn.Module:
    """cfg is expected to optionally contain a "loss_weights" dict with
    keys "l1", "gradient", "ssim". If absent, falls back to pure L1
    (backward compatible with configs that predate this loss).
    """
    weights = cfg.get("loss_weights", None)
    if weights is None:
        return _PureL1Wrapper()
    return CombinedLoss(
        w_l1=weights.get("l1", 1.0),
        w_gradient=weights.get("gradient", 0.5),
        w_ssim=weights.get("ssim", 0.2),
    )


class _PureL1Wrapper(nn.Module):
    """Wraps plain L1 so it has the same (loss, components_dict) return
    signature as CombinedLoss -- lets train.py treat both loss types
    identically without branching logic.
    """

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        return l1, {"l1": l1.item(), "total": l1.item()}


# ---------------------------------------------------------------------------
# Adversarial (GAN) losses -- LSGAN formulation
# ---------------------------------------------------------------------------
# LSGAN (Mao et al., 2017) replaces the original GAN's log-loss with a
# least-squares loss: real predictions are pushed toward 1, fake predictions
# toward 0 (discriminator) or 1 (generator wants to fool it). This is
# generally more stable to train than the original binary cross-entropy GAN
# loss, with fewer vanishing-gradient issues -- a reasonable stability choice
# given the short fine-tuning budget this project has for adversarial
# training.

def discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Discriminator wants: real_logits -> 1, fake_logits -> 0."""
    real_loss = F.mse_loss(real_logits, torch.ones_like(real_logits))
    fake_loss = F.mse_loss(fake_logits, torch.zeros_like(fake_logits))
    return 0.5 * (real_loss + fake_loss)


def generator_adversarial_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Generator wants: fake_logits (as judged by discriminator) -> 1,
    i.e. it wants the discriminator to be fooled into thinking its output
    is real.
    """
    return F.mse_loss(fake_logits, torch.ones_like(fake_logits))


class GeneratorGANLoss(nn.Module):
    """Combined generator loss for adversarial fine-tuning: pixel-level L1
    (anchors output to ground truth, prevents the generator from drifting
    into plausible-but-wrong content) + adversarial loss (pushes toward
    sharper, more realistic local texture per the discriminator's judgment).

    L1 is kept dominant (large w_l1 relative to w_adv) since this is a
    FINE-TUNING stage starting from an already pixel-loss-trained
    checkpoint, not training from scratch -- the goal is to sharpen an
    already-reasonable reconstruction, not to let adversarial loss
    completely override it.
    """

    def __init__(self, w_l1: float = 1.0, w_adv: float = 0.01):
        super().__init__()
        self.w_l1 = w_l1
        self.w_adv = w_adv

    def forward(self, pred: torch.Tensor, target: torch.Tensor, fake_logits: torch.Tensor):
        l1 = F.l1_loss(pred, target)
        adv = generator_adversarial_loss(fake_logits)
        total = self.w_l1 * l1 + self.w_adv * adv
        components = {"l1": l1.item(), "adversarial": adv.item(), "total": total.item()}
        return total, components


if __name__ == "__main__":
    # Smoke test against random tensors of a realistic shape.
    pred = torch.rand(4, 3, 128, 128, requires_grad=True)
    target = torch.rand(4, 3, 128, 128)

    combined = CombinedLoss()
    total, components = combined(pred, target)
    print("CombinedLoss components:", components)
    total.backward()
    print("Backward pass OK, pred.grad is not None:", pred.grad is not None)

    pure = _PureL1Wrapper()
    total2, components2 = pure(pred.detach().requires_grad_(), target)
    print("Pure L1 components:", components2)