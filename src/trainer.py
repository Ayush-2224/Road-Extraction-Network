"""
src/trainer.py
--------------
Full training loop for road segmentation.

Features:
  - Mixed precision (torch.cuda.amp) — essential for RTX 3050 6GB
  - AdamW optimizer + CosineAnnealingLR / ReduceLROnPlateau
  - Early stopping based on val IoU
  - Optimal threshold search after each epoch
  - TensorBoard + JSONL logging
  - Gradient clipping
  - Best model checkpointing
  - Validation grid visualisation saved every N epochs
"""

import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional

from src.metrics import MetricAccumulator, compute_metrics, find_best_threshold
from src.utils import save_checkpoint, TrainingLogger, save_prediction_grid


class EarlyStopping:
    """Stop training when monitored metric stops improving."""

    def __init__(self, patience: int = 12, min_delta: float = 1e-4, mode: str = "max"):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.best      = -float("inf") if mode == "max" else float("inf")
        self.counter   = 0
        self.stop      = False

    def step(self, value: float) -> bool:
        improved = (
            value > self.best + self.min_delta if self.mode == "max"
            else value < self.best - self.min_delta
        )
        if improved:
            self.best    = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return improved


class Trainer:
    """
    Encapsulates the full train / validate cycle.

    Usage:
        trainer = Trainer(model, cfg, device)
        trainer.fit(train_loader, val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        cfg: dict,
        device: torch.device,
        run_name: str = "run",
    ):
        self.model   = model.to(device)
        self.loss_fn = loss_fn
        self.cfg     = cfg
        self.device  = device
        self.run_name = run_name

        train_cfg = cfg.get("training", {})

        # ── Optimizer ──
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg.get("learning_rate", 1e-4),
            weight_decay=train_cfg.get("weight_decay", 1e-4),
        )

        # ── LR Scheduler ──
        sched_type = train_cfg.get("scheduler", "cosine")
        epochs     = train_cfg.get("epochs", 60)
        if sched_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=train_cfg.get("cosine_t_max", epochs),
                eta_min=1e-6,
            )
        else:  # plateau
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                patience=train_cfg.get("plateau_patience", 5),
                factor=train_cfg.get("plateau_factor", 0.5),
                verbose=True,
            )

        # ── AMP Scaler ──
        self.use_amp = train_cfg.get("amp", True) and device.type == "cuda"
        self.scaler  = GradScaler(enabled=self.use_amp)

        # ── Early Stopping ──
        self.early_stopping = EarlyStopping(
            patience=train_cfg.get("early_stopping_patience", 12),
            mode="max",
        ) if train_cfg.get("early_stopping", True) else None

        # ── Paths ──
        paths_cfg = cfg.get("paths", {})
        self.checkpoint_dir = paths_cfg.get("checkpoint_dir", "runs/checkpoints")
        self.result_dir     = paths_cfg.get("result_dir",     "results")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.result_dir,     exist_ok=True)

        # ── Config params ──
        self.epochs       = epochs
        self.grad_clip    = train_cfg.get("grad_clip", 1.0)
        self.threshold    = cfg.get("threshold", {}).get("default", 0.5)
        self.optimize_thr = cfg.get("threshold", {}).get("optimize", True)
        self.thr_range    = cfg.get("threshold", {}).get("search_range", [0.3, 0.7])
        self.thr_steps    = cfg.get("threshold", {}).get("search_steps", 9)

        # ── Logger ──
        self.logger = TrainingLogger(
            log_dir=cfg.get("paths", {}).get("log_dir", "runs/logs"),
            run_name=run_name,
        )

        print(f"\n[Trainer] {'AMP=ON' if self.use_amp else 'AMP=OFF'} | "
              f"LR={train_cfg.get('learning_rate', 1e-4)} | "
              f"Epochs={self.epochs} | "
              f"Early stopping patience={train_cfg.get('early_stopping_patience', 12)}\n")

    # ── Training Epoch ────────────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        acc = MetricAccumulator()
        bar = tqdm(loader, desc="  Train", leave=False, dynamic_ncols=True)

        for batch in bar:
            images = batch["image"].to(self.device, non_blocking=True)
            masks  = batch["mask"].to(self.device,  non_blocking=True)

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss   = self.loss_fn(logits, masks)

            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            with torch.no_grad():
                proba = torch.sigmoid(logits)
                m     = compute_metrics(proba, masks, self.threshold)
                m["loss"] = loss.item()
            acc.update(m)
            bar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{m['iou']:.4f}")

        return acc.averages()

    # ── Validation Epoch ──────────────────────────────────────────────────────

    def _val_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        collect_for_threshold: bool = True,
    ) -> Dict[str, float]:
        self.model.eval()
        acc = MetricAccumulator()
        pred_probas_all = []
        targets_all     = []
        first_batch     = None

        bar = tqdm(loader, desc="  Val  ", leave=False, dynamic_ncols=True)

        with torch.no_grad():
            for i, batch in enumerate(bar):
                images = batch["image"].to(self.device, non_blocking=True)
                masks  = batch["mask"].to(self.device,  non_blocking=True)

                with autocast(enabled=self.use_amp):
                    logits = self.model(images)
                    loss   = self.loss_fn(logits, masks)

                proba = torch.sigmoid(logits)
                m     = compute_metrics(proba, masks, self.threshold)
                m["loss"] = loss.item()
                acc.update(m)

                if collect_for_threshold:
                    pred_probas_all.append(proba.cpu().numpy())
                    targets_all.append(masks.cpu().numpy())

                if i == 0:
                    first_batch = (images[:4], masks[:4], proba[:4])

                bar.set_postfix(iou=f"{m['iou']:.4f}")

        # ── Threshold Optimisation ──
        if collect_for_threshold and self.optimize_thr:
            all_preds   = [p[j, 0] for p in pred_probas_all for j in range(p.shape[0])]
            all_targets = [t[j, 0] for t in targets_all     for j in range(t.shape[0])]
            new_thr, thr_iou = find_best_threshold(
                all_preds, all_targets,
                search_range=tuple(self.thr_range),
                steps=self.thr_steps,
                verbose=(epoch % 5 == 0),   # Print detail every 5 epochs
            )
            self.threshold = new_thr

        # ── Save validation grid every 5 epochs ──
        if first_batch is not None and epoch % 5 == 0:
            imgs, gts, preds = first_batch
            grid_path = os.path.join(self.result_dir, f"val_grid_epoch{epoch:03d}.png")
            save_prediction_grid(imgs, gts, preds, grid_path, threshold=self.threshold)

        return acc.averages()

    # ── Main Fit Loop ─────────────────────────────────────────────────────────

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        best_iou = 0.0
        start    = time.time()

        for epoch in range(1, self.epochs + 1):
            lr = self.optimizer.param_groups[0]["lr"]
            print(f"\n{'─'*60}")
            print(f"  Epoch {epoch}/{self.epochs}   LR={lr:.2e}   "
                  f"Threshold={self.threshold:.3f}")
            print(f"{'─'*60}")

            # Training
            t_metrics = self._train_epoch(train_loader)

            # Validation
            v_metrics = self._val_epoch(val_loader, epoch)

            val_iou = v_metrics.get("iou", 0.0)

            # LR scheduler step
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_iou)
            else:
                self.scheduler.step()

            # Log metrics
            all_metrics = {
                "train_loss": t_metrics.get("loss", 0),
                "train_iou":  t_metrics.get("iou",  0),
                "val_loss":   v_metrics.get("loss",  0),
                "val_iou":    v_metrics.get("iou",   0),
                "val_f1":     v_metrics.get("f1",    0),
                "val_prec":   v_metrics.get("precision", 0),
                "val_rec":    v_metrics.get("recall",    0),
                "lr":         lr,
                "threshold":  self.threshold,
            }
            self.logger.log(epoch, all_metrics)

            print(f"  TRAIN  → {MetricAccumulator().summary_str() if False else ''}"
                  f"Loss={t_metrics.get('loss',0):.4f}  IoU={t_metrics.get('iou',0):.4f}")
            print(f"  VAL    → Loss={v_metrics.get('loss',0):.4f}  "
                  f"IoU={val_iou:.4f}  F1={v_metrics.get('f1',0):.4f}  "
                  f"P={v_metrics.get('precision',0):.4f}  R={v_metrics.get('recall',0):.4f}")

            # Checkpoint
            is_best = val_iou > best_iou
            if is_best:
                best_iou = val_iou

            save_checkpoint(
                self.model, self.optimizer, epoch, all_metrics,
                threshold=self.threshold,
                cfg=self.cfg,
                save_dir=self.checkpoint_dir,
                is_best=is_best,
                tag=self.run_name,
            )

            # Early stopping
            if self.early_stopping is not None:
                self.early_stopping.step(val_iou)
                if self.early_stopping.stop:
                    print(f"\n[Early Stopping] No improvement for "
                          f"{self.early_stopping.patience} epochs. Stopping.")
                    break

        elapsed = (time.time() - start) / 60
        print(f"\n{'='*60}")
        print(f"  Training complete in {elapsed:.1f} min")
        print(f"  Best Val IoU: {best_iou:.4f}")
        print(f"  Best threshold: {self.threshold:.3f}")
        print(f"  Checkpoint: {self.checkpoint_dir}/best_model.pth")
        print(f"{'='*60}\n")

        self.logger.close()
        return best_iou
