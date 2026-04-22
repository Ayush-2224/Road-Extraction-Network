"""
src/losses.py
-------------
Combined loss functions for road segmentation.

Key design:
- Roads = very few pixels vs background → strong class imbalance
- BCE alone: biased to background
- Dice alone: can be unstable early in training
- Combined BCE + Dice: stable + imbalance-aware
- pos_weight for BCE: up-weights road class (foreground)
- FocalDiceLoss: extra focus on hard misclassified pixels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss computed on sigmoid-activated predictions.
    Works on logits (applies sigmoid internally).
    smooth prevents division by zero.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions
        probs_flat  = probs.view(probs.size(0), -1)
        target_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * target_flat).sum(dim=1)
        denom = probs_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice_per_sample = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice_per_sample.mean()


class BCEDiceLoss(nn.Module):
    """
    Weighted BCE + Dice combined loss.

    Args:
        bce_weight:        Weight for BCE component.
        dice_weight:       Weight for Dice component.
        road_class_weight: pos_weight for BCE — roads are upweighted.
                           Typical range: 2.0 – 5.0 depending on road density.
                           Default 3.0 works well for DeepGlobe / Massachusetts.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        road_class_weight: float = 3.0,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.road_class_weight = road_class_weight
        self.dice_loss = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_weight = torch.tensor(
            [self.road_class_weight], dtype=logits.dtype, device=logits.device
        )
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight
        )
        dice = self.dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


class FocalDiceLoss(nn.Module):
    """
    Focal Loss + Dice combined.
    Focal Loss down-weights easy examples and focuses training on hard cases
    (misclassified road pixels, thin roads at intersections).

    Args:
        alpha:       Weighting factor for positive class.
        gamma:       Focusing parameter. Higher = more focus on hard examples.
        focal_weight: Weight for Focal component.
        dice_weight:  Weight for Dice component.
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_per_pixel = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        # pt: probability of correct class
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        # alpha weighting per class
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal = alpha_t * (1.0 - pt) ** self.gamma * bce_per_pixel
        focal_loss = focal.mean()

        dice = self.dice_loss(logits, targets)
        return self.focal_weight * focal_loss + self.dice_weight * dice


def build_loss(cfg: dict) -> nn.Module:
    """Factory: build the loss from config dict."""
    loss_cfg = cfg.get("loss", {})
    loss_type = loss_cfg.get("type", "bce_dice")

    if loss_type == "bce_dice":
        return BCEDiceLoss(
            bce_weight=loss_cfg.get("bce_weight", 0.5),
            dice_weight=loss_cfg.get("dice_weight", 0.5),
            road_class_weight=loss_cfg.get("road_class_weight", 3.0),
        )
    elif loss_type == "focal_dice":
        return FocalDiceLoss(
            alpha=loss_cfg.get("focal_alpha", 0.75),
            gamma=loss_cfg.get("focal_gamma", 2.0),
            focal_weight=loss_cfg.get("bce_weight", 0.5),
            dice_weight=loss_cfg.get("dice_weight", 0.5),
        )
    else:
        raise ValueError(f"Unknown loss type: '{loss_type}'. Use 'bce_dice' or 'focal_dice'.")
