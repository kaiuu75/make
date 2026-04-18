"""Loss functions.

We combine three complementary signals:

1. **Weighted focal BCE** on the change head. Handles the heavy class
   imbalance (deforestation pixels are rare) and down-weights easy negatives.
2. **Soft Dice** on the change head. Directly optimises polygon/area overlap,
   which is exactly what Union IoU measures at the leaderboard level.
3. **Masked cross-entropy** on the month head, applied only on pixels where
   the weak labels agree that a change occurred. This targets Year Accuracy.

Tunables live in ``configs/server.yaml`` under ``deep.loss``.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - torch optional
    torch = None  # type: ignore
    F = None  # type: ignore


@dataclass(frozen=True)
class LossWeights:
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    dice_weight: float = 1.0
    month_ce_weight: float = 0.25


def focal_bce_with_weight(
    logits: "torch.Tensor",
    targets: "torch.Tensor",
    weight: "torch.Tensor",
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
    eps: float = 1e-6,
) -> "torch.Tensor":
    """Sigmoid-focal BCE, per-pixel weighted.

    ``logits``, ``targets`` and ``weight`` are all (B, H, W).
    """
    assert torch is not None
    p = torch.sigmoid(logits)
    p_t = torch.where(targets > 0.5, p, 1 - p).clamp(eps, 1 - eps)
    alpha_t = torch.where(targets > 0.5, alpha, 1 - alpha)
    loss = -alpha_t * (1 - p_t).pow(gamma) * torch.log(p_t)
    return (loss * weight).mean()


def soft_dice(
    logits: "torch.Tensor",
    targets: "torch.Tensor",
    forest_mask: "torch.Tensor",
    *,
    eps: float = 1e-6,
) -> "torch.Tensor":
    """Soft-Dice on the foreground class, restricted to in-forest pixels."""
    assert torch is not None
    p = torch.sigmoid(logits) * forest_mask
    t = targets * forest_mask
    inter = (p * t).sum(dim=(1, 2))
    denom = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


def month_cross_entropy(
    month_logits: "torch.Tensor",
    month_target: "torch.Tensor",
    *,
    ignore_index: int = -100,
) -> "torch.Tensor":
    """Masked cross-entropy on month-of-change.

    ``month_logits`` is (B, M, H, W), ``month_target`` is (B, H, W) long.
    """
    assert torch is not None
    if (month_target >= 0).sum() == 0:
        return month_logits.new_zeros(())
    return F.cross_entropy(month_logits, month_target, ignore_index=ignore_index)


def total_loss(
    outputs: dict[str, "torch.Tensor"],
    batch: dict[str, "torch.Tensor"],
    weights: LossWeights,
) -> tuple["torch.Tensor", dict[str, float]]:
    change_logits = outputs["change_logits"]
    month_logits = outputs["month_logits"]

    focal = focal_bce_with_weight(
        change_logits,
        batch["y_change"],
        batch["weight"],
        alpha=weights.focal_alpha,
        gamma=weights.focal_gamma,
    )
    dice = soft_dice(change_logits, batch["y_change"], batch["forest"])
    ce = month_cross_entropy(month_logits, batch["y_month"])

    loss = focal + weights.dice_weight * dice + weights.month_ce_weight * ce
    logs = {
        "loss": float(loss.detach()),
        "focal": float(focal.detach()),
        "dice": float(dice.detach()),
        "month_ce": float(ce.detach()),
    }
    return loss, logs
