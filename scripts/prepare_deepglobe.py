"""
scripts/prepare_deepglobe.py
-----------------------------
Preprocess the DeepGlobe Road Extraction Dataset.

DeepGlobe raw format:
  <data_dir>/
    ├── {id}_sat.jpg   ← RGB satellite image (1024×1024)
    └── {id}_mask.png  ← RGB color mask  (road = [255,255,255] or white)

This script:
  1. Reads each image + mask pair
  2. Converts mask to binary (road vs background)
  3. Tiles into 512×512 patches with configurable stride
  4. Applies a minimum road coverage filter (skip near-empty patches)
  5. Splits into train / val / test sets
  6. Saves to data/processed/{split}/images/ and data/processed/{split}/masks/

Usage:
  python scripts/prepare_deepglobe.py --data-dir data/datasets/deepglobe
  python scripts/prepare_deepglobe.py --data-dir data/datasets/deepglobe --patch-size 512 --stride 384
"""

import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",    required=True, help="Raw DeepGlobe directory")
    p.add_argument("--output-dir",  default="data/processed", help="Output base directory")
    p.add_argument("--patch-size",  type=int, default=512)
    p.add_argument("--stride",      type=int, default=384,
                   help="Patch stride. Smaller = more overlap = more patches (default 384 = 25% overlap)")
    p.add_argument("--min-road-ratio", type=float, default=0.01,
                   help="Minimum fraction of road pixels to keep a patch (default 1%)")
    p.add_argument("--val-ratio",   type=float, default=0.15)
    p.add_argument("--test-ratio",  type=float, default=0.10)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def extract_patches(image: np.ndarray, mask: np.ndarray, patch_size: int, stride: int):
    """Yield (patch_img, patch_mask) tuples using sliding window."""
    H, W = image.shape[:2]
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            yield (
                image[y:y+patch_size, x:x+patch_size],
                mask [y:y+patch_size, x:x+patch_size],
            )
    # Include right/bottom edge patches
    if H > patch_size:
        y = H - patch_size
        for x in range(0, W - patch_size + 1, stride):
            yield (image[y:y+patch_size, x:x+patch_size], mask[y:y+patch_size, x:x+patch_size])
    if W > patch_size:
        x = W - patch_size
        for y in range(0, H - patch_size + 1, stride):
            yield (image[y:y+patch_size, x:x+patch_size], mask[y:y+patch_size, x:x+patch_size])


def main():
    args = parse_args()
    random.seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.output_dir)

    # Find all sat images
    sat_files = sorted(data_dir.rglob("*_sat.jpg"))
    if not sat_files:
        print(f"[ERROR] No *_sat.jpg files found in '{data_dir}'")
        print("  Make sure DeepGlobe data is in the correct directory.")
        return

    print(f"Found {len(sat_files)} DeepGlobe image pairs in '{data_dir}'")

    # Split file-level into train/val/test (before tiling to avoid data leakage)
    random.shuffle(sat_files)
    n_total = len(sat_files)
    n_test  = max(1, int(n_total * args.test_ratio))
    n_val   = max(1, int(n_total * args.val_ratio))
    test_files  = sat_files[:n_test]
    val_files   = sat_files[n_test:n_test + n_val]
    train_files = sat_files[n_test + n_val:]

    splits = [("train", train_files), ("val", val_files), ("test", test_files)]

    total_patches = 0
    skipped       = 0

    for split_name, files in splits:
        img_out = out_dir / split_name / "images"
        msk_out = out_dir / split_name / "masks"
        img_out.mkdir(parents=True, exist_ok=True)
        msk_out.mkdir(parents=True, exist_ok=True)

        split_count = 0
        print(f"\n[{split_name.upper()}] Processing {len(files)} images ...")

        for sat_file in tqdm(files, desc=f"  {split_name}"):
            img_id   = sat_file.stem.replace("_sat", "")
            mask_file = sat_file.parent / f"{img_id}_mask.png"

            if not mask_file.exists():
                print(f"  [SKIP] Mask not found for {img_id}")
                continue

            # Load image
            image = cv2.imread(str(sat_file), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Load + binarize mask
            mask_raw = cv2.imread(str(mask_file), cv2.IMREAD_COLOR)
            # DeepGlobe road mask: roads are white [255,255,255]
            # Some versions use a color encoding — we threshold the green channel
            mask_gray = cv2.cvtColor(mask_raw, cv2.COLOR_BGR2GRAY)
            mask_bin  = (mask_gray > 127).astype(np.uint8) * 255

            # Tile
            for patch_idx, (p_img, p_mask) in enumerate(
                extract_patches(image, mask_bin, args.patch_size, args.stride)
            ):
                road_ratio = (p_mask > 127).mean()
                if road_ratio < args.min_road_ratio:
                    skipped += 1
                    continue

                patch_name = f"{img_id}_p{patch_idx:04d}"
                cv2.imwrite(str(img_out / f"{patch_name}.jpg"),
                            cv2.cvtColor(p_img, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(msk_out / f"{patch_name}.png"), p_mask)
                split_count += 1

        print(f"  → {split_count} patches saved")
        total_patches += split_count

    print(f"\n{'='*50}")
    print(f"  Total patches saved : {total_patches}")
    print(f"  Patches skipped     : {skipped} (< {args.min_road_ratio*100:.1f}% road coverage)")
    print(f"  Output directory    : {out_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
