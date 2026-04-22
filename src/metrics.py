"""
src/metrics.py
--------------
Evaluation metrics for road segmentation.

Metrics:
  - IoU (primary metric)
  - F1 / Dice Coefficient
  - Precision
  - Recall

Threshold optimization:
  - Sweep threshold in [0.3, 0.7] range on validation predictions
  - Select threshold that maximises mean IoU
  - Saved to checkpoint for use at inference time
"""

import numpy as np
import torch
from typing import Tuple, Dict, List


# ──────────────────────────────────────────────────────────
#  Core Metric Functions
# ──────────────────────────────────────────────────────────

def compute_iou(
    pred_binary: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """Intersection over Union (Jaccard Index)."""
    intersection = (pred_binary * target).sum()
    union = (pred_binary + target).clamp(0, 1).sum()
    return ((intersection + eps) / (union + eps)).item()


def compute_precision_recall(
    pred_binary: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[float, float]:
    """
    Precision and Recall.
    Important for road continuity:
      - High Recall  → fewer missed roads
      - High Precision → fewer false roads
    """
    tp = (pred_binary * target).sum()
    fp = (pred_binary * (1.0 - target)).sum()
    fn = ((1.0 - pred_binary) * target).sum()

    precision = ((tp + eps) / (tp + fp + eps)).item()
    recall    = ((tp + eps) / (tp + fn + eps)).item()
    return precision, recall


def compute_f1(
    pred_binary: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """F1 / Dice Coefficient."""
    precision, recall = compute_precision_recall(pred_binary, target, eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return f1


def compute_metrics(
    pred_proba: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all metrics at a given threshold.

    Args:
        pred_proba: [B, 1, H, W] or [H, W] probability map (after sigmoid).
        target:     [B, 1, H, W] or [H, W] binary ground truth.
        threshold:  Binarization threshold.

    Returns:
        dict with keys: iou, f1, precision, recall
    """
    pred_binary = (pred_proba > threshold).float()
    iou = compute_iou(pred_binary, target)
    precision, recall = compute_precision_recall(pred_binary, target)
    f1  = 2 * precision * recall / (precision + recall + 1e-6)
    return {"iou": iou, "f1": f1, "precision": precision, "recall": recall}


# ──────────────────────────────────────────────────────────
#  Optimal Threshold Search
# ──────────────────────────────────────────────────────────

def find_best_threshold(
    pred_probas: List[np.ndarray],
    targets: List[np.ndarray],
    search_range: Tuple[float, float] = (0.3, 0.7),
    steps: int = 9,
    verbose: bool = True,
) -> Tuple[float, float]:
    """
    Sweep threshold values, return the one maximising mean IoU over the
    validation set.

    Args:
        pred_probas: List of probability arrays (H, W) in [0, 1].
        targets:     Matching list of binary ground truth arrays (H, W).
        search_range: Inclusive [low, high] range to search.
        steps:        Number of threshold values to test.
        verbose:      Print results for each threshold.

    Returns:
        (best_threshold, best_mean_iou)
    """
    thresholds = np.linspace(search_range[0], search_range[1], steps)
    best_threshold = 0.5
    best_iou = 0.0

    if verbose:
        print("\n[Threshold Search]")
        print(f"{'Threshold':>12} | {'Mean IoU':>10}")
        print("-" * 27)

    for thresh in thresholds:
        ious = []
        for pred, tgt in zip(pred_probas, targets):
            pred_t = torch.tensor(pred, dtype=torch.float32)
            tgt_t  = torch.tensor(tgt,  dtype=torch.float32)
            pred_bin = (pred_t > thresh).float()
            ious.append(compute_iou(pred_bin, tgt_t))

        mean_iou = float(np.mean(ious))
        if verbose:
            marker = " ◀ best" if mean_iou > best_iou else ""
            print(f"{thresh:>12.3f} | {mean_iou:>10.4f}{marker}")

        if mean_iou > best_iou:
            best_iou = mean_iou
            best_threshold = float(thresh)

    if verbose:
        print(f"\n→ Best threshold: {best_threshold:.3f}  (IoU = {best_iou:.4f})\n")

    return best_threshold, best_iou


# ──────────────────────────────────────────────────────────
#  Metrics Accumulator (for epoch-level averaging)
# ──────────────────────────────────────────────────────────

class MetricAccumulator:
    """Accumulate per-batch metrics and compute epoch averages."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._metrics: Dict[str, List[float]] = {
            "iou": [], "f1": [], "precision": [], "recall": [], "loss": []
        }

    def update(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            if k in self._metrics:
                self._metrics[k].append(v)

    def averages(self) -> Dict[str, float]:
        return {k: float(np.mean(v)) if v else 0.0 for k, v in self._metrics.items()}

    def summary_str(self) -> str:
        avgs = self.averages()
        return (
            f"Loss={avgs.get('loss', 0):.4f} | "
            f"IoU={avgs.get('iou', 0):.4f} | "
            f"F1={avgs.get('f1', 0):.4f} | "
            f"P={avgs.get('precision', 0):.4f} | "
            f"R={avgs.get('recall', 0):.4f}"
        )
