"""
src/augmentations.py
--------------------
Albumentations-based augmentation pipelines.
All transforms are applied consistently to both image AND mask (same random seed).
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet normalization stats
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_train_transforms(cfg: dict) -> A.Compose:
    """
    Full augmentation pipeline for training.
    Road-structure-safe: spatial transforms are applied consistently to mask.
    """
    aug_cfg = cfg.get("augmentation", {})
    transforms = [
        # --- Spatial ---
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ]

    if aug_cfg.get("use_elastic", True):
        transforms.append(
            A.ElasticTransform(
                alpha=aug_cfg.get("elastic_alpha", 120),
                sigma=aug_cfg.get("elastic_sigma", 6.0),
                alpha_affine=120 * 0.03,
                border_mode=0,
                p=0.3,
            )
        )

    if aug_cfg.get("use_grid_distortion", False):
        transforms.append(A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2))

    # --- Photometric (image only, mask unaffected) ---
    transforms += [
        A.RandomBrightnessContrast(
            brightness_limit=aug_cfg.get("brightness_limit", 0.3),
            contrast_limit=aug_cfg.get("contrast_limit", 0.3),
            p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=aug_cfg.get("hue_shift", 20),
            sat_shift_limit=aug_cfg.get("sat_shift", 30),
            val_shift_limit=20,
            p=0.3,
        ),
        A.GaussNoise(
            var_limit=tuple(aug_cfg.get("gauss_noise_var", [10, 50])),
            p=0.3,
        ),
        A.Blur(
            blur_limit=aug_cfg.get("blur_limit", 3),
            p=0.2,
        ),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        # Coarse dropout — randomly masks small patches (forces model to use context)
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            fill_value=0,
            p=0.2,
        ),
    ]

    # --- Normalize + ToTensor (ALWAYS last) ---
    transforms += [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ]

    return A.Compose(transforms)


def get_val_transforms(cfg: dict = None) -> A.Compose:
    """Minimal transforms for validation and test: normalize + to tensor only."""
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ])


def get_tta_transforms() -> list:
    """
    Test-Time Augmentation variants.
    Returns list of (transform, inverse_transform) tuples.
    Average all predictions for the final result.
    """
    base = A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return [base, base, base, base]
