"""
src/utils.py
------------
Shared utilities: config loading, checkpointing, visualization, logging.
"""

import os
import json
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (server-safe)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


# ──────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Load YAML config and return as nested dict."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def merge_config(base_cfg: dict, overrides: dict) -> dict:
    """Recursively merge override dict into base config."""
    cfg = base_cfg.copy()
    for k, v in overrides.items():
        if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
            cfg[k] = merge_config(cfg[k], v)
        else:
            cfg[k] = v
    return cfg


# ──────────────────────────────────────────────────────────
#  Device
# ──────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Device] CUDA — {name} ({vram:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("[Device] CPU only — training will be slow!")
    return device


# ──────────────────────────────────────────────────────────
#  Model Factory
# ──────────────────────────────────────────────────────────

def build_model(cfg: dict) -> torch.nn.Module:
    """Instantiate model from config."""
    model_cfg = cfg["model"]
    arch      = model_cfg.get("architecture", "unet")
    encoder   = model_cfg.get("encoder", "resnet50")
    pretrained = model_cfg.get("pretrained", True)
    in_ch     = model_cfg.get("in_channels", 3)
    num_cls   = model_cfg.get("num_classes", 1)

    if arch == "unet":
        from src.models.unet import build_unet
        return build_unet(encoder, pretrained, in_ch, num_cls)
    elif arch == "attention_unet":
        from src.models.attention_unet import build_attention_unet
        return build_attention_unet(encoder, pretrained, in_ch, num_cls)
    elif arch == "deeplabv3plus":
        from src.models.deeplabv3plus import build_deeplabv3plus
        return build_deeplabv3plus(encoder, pretrained, in_ch, num_cls)
    else:
        raise ValueError(f"Unknown architecture: '{arch}'. "
                         f"Use 'unet', 'attention_unet', or 'deeplabv3plus'.")


# ──────────────────────────────────────────────────────────
#  Checkpointing
# ──────────────────────────────────────────────────────────

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    threshold: float,
    cfg: dict,
    save_dir: str,
    is_best: bool = False,
    tag: str = "checkpoint",
):
    """Save model checkpoint with metadata."""
    os.makedirs(save_dir, exist_ok=True)

    state = {
        "epoch":      epoch,
        "state_dict": model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "metrics":    metrics,
        "threshold":  threshold,    # Optimised threshold saved with checkpoint
        "config":     cfg,
        "timestamp":  datetime.now().isoformat(),
    }

    filename = f"{tag}_latest.pth"
    torch.save(state, os.path.join(save_dir, filename))

    if is_best:
        best_path = os.path.join(save_dir, "best_model.pth")
        torch.save(state, best_path)
        print(f"  ✓ Best model saved → {best_path}  (IoU={metrics.get('val_iou', 0):.4f})")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict:
    """Load checkpoint; return metadata dict."""
    if device is None:
        device = torch.device("cpu")
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    print(f"[Checkpoint] Loaded from '{checkpoint_path}' "
          f"(epoch {state.get('epoch', '?')}, "
          f"IoU={state.get('metrics', {}).get('val_iou', '?')})")
    return state


# ──────────────────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────────────────

class TrainingLogger:
    """Writes per-epoch metrics to a JSON Lines file + TensorBoard."""

    def __init__(self, log_dir: str, run_name: str = "run"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"{run_name}_metrics.jsonl")
        self.writer   = None

        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))
            print(f"[TensorBoard] Logging to '{log_dir}/{run_name}'")
        except ImportError:
            print("[TensorBoard] Not available — using JSONL log only.")

    def log(self, epoch: int, metrics: Dict[str, float]):
        entry = {"epoch": epoch, **metrics, "ts": datetime.now().isoformat()}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        if self.writer:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(k, v, epoch)

    def close(self):
        if self.writer:
            self.writer.close()


# ──────────────────────────────────────────────────────────
#  Visualization
# ──────────────────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized [C,H,W] tensor → uint8 RGB [H,W,C] array."""
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def save_prediction_grid(
    images: torch.Tensor,
    masks_gt: torch.Tensor,
    masks_pred: torch.Tensor,
    save_path: str,
    n_samples: int = 4,
    threshold: float = 0.5,
):
    """
    Save a grid showing: Image | Ground Truth | Prediction | Overlay.

    Args:
        images:      [B, C, H, W] normalized image tensors.
        masks_gt:    [B, 1, H, W] ground truth binary masks.
        masks_pred:  [B, 1, H, W] predicted probability maps.
        save_path:   Output PNG path.
        n_samples:   Number of samples to show.
        threshold:   Binarization threshold for predictions.
    """
    n = min(n_samples, images.shape[0])
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Satellite Image", "Ground Truth", "Prediction", "Overlay"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold")

    for i in range(n):
        img  = denormalize(images[i])
        gt   = masks_gt[i, 0].cpu().numpy()
        pred = masks_pred[i, 0].cpu().numpy()
        pred_bin = (pred > threshold).astype(np.uint8)

        # Col 0: Image
        axes[i, 0].imshow(img)
        # Col 1: Ground Truth
        axes[i, 1].imshow(gt, cmap="Blues", vmin=0, vmax=1)
        # Col 2: Prediction
        axes[i, 2].imshow(pred_bin, cmap="Blues", vmin=0, vmax=1)
        # Col 3: Overlay — roads in cyan on image
        overlay = img.copy()
        road_px = pred_bin.astype(bool)
        overlay[road_px] = np.clip(
            overlay[road_px].astype(np.int32) + np.array([0, 80, 120]), 0, 255
        ).astype(np.uint8)
        axes[i, 3].imshow(overlay)

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_overlay(
    image: np.ndarray,
    road_mask: np.ndarray,
    save_path: str,
    road_color: tuple = (0, 120, 255),
    alpha: float = 0.45,
):
    """
    Save a single overlay image: satellite image with predicted roads coloured.

    Args:
        image:      uint8 RGB [H, W, 3] satellite image.
        road_mask:  uint8 {0, 255} [H, W] binary road mask.
        save_path:  Output path (.png or .jpg).
        road_color: BGR colour for road overlay.
        alpha:      Blend factor for road highlight.
    """
    import cv2
    overlay = image.copy()
    road_layer = np.zeros_like(image)
    road_layer[road_mask > 0] = road_color[::-1]  # RGB → BGR for cv2

    cv2.addWeighted(road_layer, alpha, overlay, 1 - alpha, 0, overlay)
    # Draw road edges for clarity
    edges = cv2.Canny(road_mask, 50, 150)
    overlay[edges > 0] = [255, 255, 255]

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def plot_training_curves(log_path: str, save_path: str):
    """Read JSONL log and plot loss + metric curves."""
    import json
    records = []
    with open(log_path) as f:
        for line in f:
            records.append(json.loads(line))

    epochs       = [r["epoch"] for r in records]
    train_losses = [r.get("train_loss",  None) for r in records]
    val_ious     = [r.get("val_iou",     None) for r in records]
    val_f1s      = [r.get("val_f1",      None) for r in records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, label="Train Loss", color="#e74c3c", linewidth=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Training Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, val_ious, label="Val IoU",  color="#3498db", linewidth=2)
    ax2.plot(epochs, val_f1s, label="Val F1",   color="#2ecc71", linewidth=2, linestyle="--")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score"); ax2.set_title("Validation Metrics")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("Training Progress", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Training curves saved → {save_path}")
