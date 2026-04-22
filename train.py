"""
train.py
--------
Main training entrypoint.

Usage examples:
  # Default (U-Net, ResNet50, DeepGlobe data)
  python train.py

  # Custom config path
  python train.py --config configs/config.yaml

  # Override architecture / encoder
  python train.py --model attention_unet --encoder resnet50

  # Override batch size (e.g. if OOM)
  python train.py --batch-size 2

  # Change loss type
  python train.py --loss focal_dice

  # Dry run (1 epoch, 1 batch) to verify everything loads
  python train.py --dry-run
"""

import argparse
import os
import random
import numpy as np
import torch

from src.utils     import load_config, build_model, get_device, merge_config
from src.dataset   import RoadDataset, DeepGlobeDataset, build_dataloader
from src.losses    import build_loss
from src.augmentations import get_train_transforms, get_val_transforms
from src.trainer   import Trainer


# ──────────────────────────────────────────────────────────
#  Reproducibility
# ──────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ──────────────────────────────────────────────────────────
#  CLI Arguments
# ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train road segmentation model")
    p.add_argument("--config",     default="configs/config.yaml", help="Config YAML path")
    p.add_argument("--model",      default=None, help="Override model architecture")
    p.add_argument("--encoder",    default=None, help="Override encoder backbone")
    p.add_argument("--epochs",     type=int, default=None, help="Override number of epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    p.add_argument("--lr",         type=float, default=None, help="Override learning rate")
    p.add_argument("--loss",       default=None, help="Override loss type (bce_dice|focal_dice)")
    p.add_argument("--run-name",   default=None, help="Run identifier (used in checkpoint names)")
    p.add_argument("--resume",     default=None, help="Path to checkpoint to resume from")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--dry-run",    action="store_true", help="1 epoch, 2 batches — verify setup")
    return p.parse_args()


# ──────────────────────────────────────────────────────────
#  Dataset Builder
# ──────────────────────────────────────────────────────────

def build_datasets(cfg: dict):
    """
    Build train + val datasets.
    Tries DeepGlobe format first; falls back to generic image/mask folders.
    """
    data_cfg   = cfg["data"]
    train_img  = data_cfg.get("train_image_dir", "data/processed/train/images")
    train_msk  = data_cfg.get("train_mask_dir",  "data/processed/train/masks")
    val_img    = data_cfg.get("val_image_dir",   "data/processed/val/images")
    val_msk    = data_cfg.get("val_mask_dir",    "data/processed/val/masks")

    train_tf = get_train_transforms(cfg)
    val_tf   = get_val_transforms(cfg)

    # Check if DeepGlobe raw format (single dir with *_sat.jpg)
    if os.path.isdir(train_img) and any(
        f.endswith("_sat.jpg") for f in os.listdir(train_img)
    ):
        train_ds = DeepGlobeDataset(train_img, transform=train_tf, mode="train")
        val_ds   = DeepGlobeDataset(val_img,   transform=val_tf,   mode="val")
    else:
        train_ds = RoadDataset(train_img, train_msk, transform=train_tf, mode="train")
        val_ds   = RoadDataset(val_img,   val_msk,   transform=val_tf,   mode="val")

    return train_ds, val_ds


# ──────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    # ── Load & merge config ──
    cfg = load_config(args.config)

    overrides = {}
    if args.model:      overrides.setdefault("model", {})["architecture"] = args.model
    if args.encoder:    overrides.setdefault("model", {})["encoder"]      = args.encoder
    if args.epochs:     overrides.setdefault("training", {})["epochs"]    = args.epochs
    if args.batch_size: overrides.setdefault("training", {})["batch_size"]= args.batch_size
    if args.lr:         overrides.setdefault("training", {})["learning_rate"] = args.lr
    if args.loss:       overrides.setdefault("loss", {})["type"]         = args.loss
    if overrides:
        cfg = merge_config(cfg, overrides)

    if args.dry_run:
        cfg["training"]["epochs"]     = 1
        cfg["training"]["batch_size"] = 2
        cfg["threshold"]["optimize"]  = False
        print("\n[DRY RUN] 1 epoch, 2 batches. Verifying pipeline only.\n")

    # ── Run name ──
    run_name = args.run_name or (
        f"{cfg['model']['architecture']}"
        f"_{cfg['model']['encoder']}"
        f"_{cfg['loss']['type']}"
    )
    print(f"\n{'='*60}")
    print(f"  Run: {run_name}")
    print(f"  Model: {cfg['model']['architecture']}  |  Encoder: {cfg['model']['encoder']}")
    print(f"  Loss: {cfg['loss']['type']}  |  Epochs: {cfg['training']['epochs']}")
    print(f"  Batch: {cfg['training']['batch_size']}  |  LR: {cfg['training']['learning_rate']}")
    print(f"{'='*60}\n")

    device = get_device()

    # ── Datasets & loaders ──
    train_ds, val_ds = build_datasets(cfg)

    train_cfg = cfg.get("training",  {})
    data_cfg  = cfg.get("data",      {})

    train_loader = build_dataloader(
        train_ds,
        batch_size=train_cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        use_urban_oversampling=data_cfg.get("urban_oversample", True),
        urban_weight_scale=data_cfg.get("urban_weight_scale", 4.0),
    )
    val_loader = build_dataloader(
        val_ds,
        batch_size=train_cfg.get("batch_size", 4),
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        use_urban_oversampling=False,
    )

    # ── Model ──
    model = build_model(cfg)

    # ── Resume ──
    if args.resume:
        from src.utils import load_checkpoint
        load_checkpoint(args.resume, model, device=device)

    # ── Loss ──
    loss_fn = build_loss(cfg)
    print(f"[Loss] {loss_fn.__class__.__name__}  "
          f"(road_class_weight={cfg['loss'].get('road_class_weight', 3.0)})")

    # ── Dry run: verify 1 batch ──
    if args.dry_run:
        model = model.to(device)
        batch = next(iter(train_loader))
        imgs  = batch["image"].to(device)
        msks  = batch["mask"].to(device)
        with torch.no_grad():
            out = model(imgs)
        loss = loss_fn(out, msks)
        print(f"[DRY RUN] Forward pass OK  |  Output shape: {out.shape}  "
              f"|  Loss: {loss.item():.4f}")
        print("[DRY RUN] All checks passed! Run without --dry-run to train.")
        return

    # ── Train ──
    trainer = Trainer(model, loss_fn, cfg, device, run_name=run_name)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
