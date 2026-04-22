"""
src/dataset.py
--------------
PyTorch Dataset classes for road segmentation.

Supports:
  - DeepGlobe format: {id}_sat.jpg / {id}_mask.png
  - Massachusetts format: {id}.tif / {id}.tif (separate mask folder)
  - Generic: images/*.{jpg,png,tif} + masks/*.{jpg,png,tif}

Urban Oversampling:
  - Road density (fraction of road pixels) is computed per patch.
  - WeightedRandomSampler uses this to oversample high-density (urban) tiles.
  - Urban patches are up to (1 + urban_weight_scale)x more likely to be sampled.
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Optional, List, Callable, Dict


class RoadDataset(Dataset):
    """
    Generic road segmentation dataset.
    Handles multiple image formats and both training and inference modes.
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        image_ids: Optional[List[str]] = None,
        mode: str = "train",   # "train" | "val" | "test" | "infer"
    ):
        """
        Args:
            image_dir:  Directory containing satellite images.
            mask_dir:   Directory containing binary road masks.
                        If None, dataset operates in inference mode (no masks).
            transform:  Albumentations Compose transform.
            image_ids:  Explicit list of stem IDs. Auto-discovered if None.
            mode:       Dataset mode string (used for logging).
        """
        self.image_dir = Path(image_dir)
        self.mask_dir  = Path(mask_dir) if mask_dir else None
        self.transform = transform
        self.mode = mode

        if image_ids is not None:
            self.image_ids = image_ids
        else:
            self.image_ids = self._discover_ids()

        print(f"[{mode.upper()} Dataset] {len(self.image_ids)} samples  "
              f"from '{self.image_dir.name}'")

    # ── Discovery ────────────────────────────────────────────────────────────

    def _discover_ids(self) -> List[str]:
        """Auto-discover image IDs from supported extensions."""
        extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
        ids = []
        for ext in extensions:
            ids += [p.stem for p in self.image_dir.glob(f"*{ext}")]
        # Deduplicate and sort deterministically
        return sorted(set(ids))

    # ── Image / Mask Loading ─────────────────────────────────────────────────

    def _load_image(self, img_id: str) -> np.ndarray:
        for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
            path = self.image_dir / f"{img_id}{ext}"
            if path.exists():
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raise FileNotFoundError(
            f"Image not found for id '{img_id}' in '{self.image_dir}'"
        )

    def _load_mask(self, img_id: str) -> np.ndarray:
        """Load and binarise road mask. Returns float32 array in {0, 1}."""
        for ext in [".png", ".jpg", ".tif", ".tiff"]:
            path = self.mask_dir / f"{img_id}{ext}"
            if path.exists():
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    return (mask > 127).astype(np.float32)
        raise FileNotFoundError(
            f"Mask not found for id '{img_id}' in '{self.mask_dir}'"
        )

    # ── Dataset Protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict:
        img_id = self.image_ids[idx]
        image  = self._load_image(img_id)

        sample: Dict = {"id": img_id}

        if self.mask_dir is not None:
            mask = self._load_mask(img_id)
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                sample["image"] = augmented["image"]          # [C, H, W] tensor
                sample["mask"]  = augmented["mask"].unsqueeze(0).float()  # [1, H, W]
            else:
                sample["image"] = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0
                sample["mask"]  = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        else:
            # Inference mode — no mask
            if self.transform:
                augmented = self.transform(image=image, mask=np.zeros(image.shape[:2], dtype=np.float32))
                sample["image"] = augmented["image"]
            else:
                sample["image"] = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0

        return sample

    # ── Urban Oversampling ────────────────────────────────────────────────────

    def compute_urban_weights(self, urban_weight_scale: float = 4.0) -> List[float]:
        """
        Compute per-sample sampling weights proportional to road density.
        Patches with more roads get higher weight → oversample urban areas.

        Args:
            urban_weight_scale: Max additional multiplier for densest patches.
        Returns:
            List of weights aligned with self.image_ids.
        """
        if self.mask_dir is None:
            return [1.0] * len(self.image_ids)

        print(f"[Urban Oversampling] Computing road densities for {len(self.image_ids)} patches...")
        weights = []
        for img_id in self.image_ids:
            try:
                mask = self._load_mask(img_id)
                density = float(mask.mean())          # Fraction of road pixels
                # Linear scale: weight = 1 + scale * density
                weight = 1.0 + urban_weight_scale * density
            except FileNotFoundError:
                weight = 1.0
            weights.append(weight)

        min_w, max_w = min(weights), max(weights)
        print(f"  -> Weight range: [{min_w:.3f}, {max_w:.3f}]  "
              f"(mean={float(np.mean(weights)):.3f})")
        return weights


# ──────────────────────────────────────────────────────────
#  DeepGlobe-specific Dataset
# ──────────────────────────────────────────────────────────

class DeepGlobeDataset(RoadDataset):
    """
    DeepGlobe Road Extraction Dataset.
    Format: {id}_sat.jpg (image) and {id}_mask.png (mask) — SAME directory.
    """

    def __init__(self, data_dir: str, transform=None, mode="train"):
        data_path = Path(data_dir)
        # Discover by sat images
        image_ids = [p.stem.replace("_sat", "")
                     for p in data_path.glob("*_sat.jpg")]
        super().__init__(
            image_dir=data_dir,
            mask_dir=data_dir,
            transform=transform,
            image_ids=image_ids,
            mode=mode,
        )

    def _load_image(self, img_id: str) -> np.ndarray:
        path = self.image_dir / f"{img_id}_sat.jpg"
        img  = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"DeepGlobe image not found: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_mask(self, img_id: str) -> np.ndarray:
        path = self.mask_dir / f"{img_id}_mask.png"
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"DeepGlobe mask not found: {path}")
        # DeepGlobe road mask: road = white (255), background = black (0)
        return (mask > 127).astype(np.float32)


# ──────────────────────────────────────────────────────────
#  DataLoader Factory
# ──────────────────────────────────────────────────────────

def build_dataloader(
    dataset: RoadDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    use_urban_oversampling: bool = False,
    urban_weight_scale: float = 4.0,
) -> DataLoader:
    """
    Build a DataLoader with optional urban oversampling via WeightedRandomSampler.
    """
    sampler = None
    if use_urban_oversampling and shuffle:
        weights = dataset.compute_urban_weights(urban_weight_scale)
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
        shuffle = False  # Sampler and shuffle are mutually exclusive

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(shuffle or sampler is not None),
    )
