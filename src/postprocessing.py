"""
src/postprocessing.py
---------------------
Post-processing pipeline applied to raw probability maps at inference time.

Urban-optimised steps:
  1. Gaussian blur   — mild smoothing, removes salt-and-pepper noise
  2. Thresholding    — binarise with optimised threshold
  3. Morphological closing — bridge road gaps (CRITICAL for urban connectivity)
  4. Dilation        — restore road width after closing
  5. Small component removal — eliminate isolated noise blobs
  6. Optional skeletonisation — extract road centrelines

Why morphological closing is critical for urban areas:
  Intersections and tree canopy create small gaps in predicted roads.
  Closing (dilation → erosion) fills these gaps without enlarging endpoints.
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
from typing import Optional


def gaussian_blur(prob_map: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Mild Gaussian smoothing on probability map before thresholding."""
    if kernel_size > 1:
        k = kernel_size | 1  # Ensure odd
        return cv2.GaussianBlur(prob_map, (k, k), sigmaX=0)
    return prob_map


def morphological_closing(
    binary_mask: np.ndarray,
    kernel_size: int = 7,
    iterations: int = 2,
) -> np.ndarray:
    """
    Morphological closing (dilation → erosion).
    Fills small dark holes and gaps in road segments.
    Larger kernel = more gap filling = better urban connectivity.

    Args:
        binary_mask:  uint8 mask in {0, 255}.
        kernel_size:  Structuring element size. Urban: 7-11.
        iterations:   Number of closing passes.
    Returns:
        Closed binary mask.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def dilate_roads(
    binary_mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """
    Slight dilation to restore road width that may shrink after other ops.
    Keeps roads connected at narrow junctions.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.dilate(binary_mask, kernel, iterations=iterations)


def remove_small_components(
    binary_mask: np.ndarray,
    min_size: int = 500,
) -> np.ndarray:
    """
    Remove connected components smaller than min_size pixels.
    Eliminates isolated noise blobs that are not roads.

    Args:
        binary_mask: uint8 mask in {0, 255}.
        min_size:    Minimum pixel count to keep a component.
    Returns:
        Cleaned binary mask.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    out = np.zeros_like(binary_mask)
    for label_idx in range(1, num_labels):  # 0 = background
        if stats[label_idx, cv2.CC_STAT_AREA] >= min_size:
            out[labels == label_idx] = 255
    return out


def extract_skeleton(binary_mask: np.ndarray, dilate_width: int = 3) -> np.ndarray:
    """
    Skeletonise roads to produce 1-pixel-wide centrelines, then re-dilate.

    Args:
        binary_mask:  uint8 mask in {0, 255}.
        dilate_width: Width to restore after skeletonisation.
    Returns:
        Centreline mask (dilated for visibility) in {0, 255}.
    """
    bool_mask = binary_mask > 0
    skeleton  = skeletonize(bool_mask).astype(np.uint8) * 255
    if dilate_width > 1:
        kernel  = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_width, dilate_width)
        )
        skeleton = cv2.dilate(skeleton, kernel, iterations=1)
    return skeleton


def postprocess(
    prob_map: np.ndarray,
    threshold: float = 0.5,
    blur_kernel: int = 3,
    closing_kernel_size: int = 7,
    closing_iterations: int = 2,
    dilation_kernel_size: int = 3,
    dilation_iterations: int = 1,
    min_component_size: int = 500,
    skeletonize_output: bool = False,
) -> np.ndarray:
    """
    Full post-processing pipeline.

    Args:
        prob_map:             float32 probability map [H, W] in [0, 1].
        threshold:            Binarization threshold (use val-optimised value).
        blur_kernel:          Gaussian blur kernel size before thresholding.
        closing_kernel_size:  Morphological closing element size.
        closing_iterations:   Number of closing passes.
        dilation_kernel_size: Dilation element size.
        dilation_iterations:  Number of dilation passes.
        min_component_size:   Pixel threshold for noise removal.
        skeletonize_output:   If True, return centreline mask.

    Returns:
        Binary mask as uint8 {0, 255} numpy array [H, W].
    """
    # Step 1: Blur
    blurred = gaussian_blur(prob_map.astype(np.float32), blur_kernel)

    # Step 2: Threshold
    binary = (blurred > threshold).astype(np.uint8) * 255

    # Step 3: Closing — fill road gaps (critical for urban connectivity)
    binary = morphological_closing(binary, closing_kernel_size, closing_iterations)

    # Step 4: Noise removal
    binary = remove_small_components(binary, min_component_size)

    # Step 5: Dilation — restore road width
    binary = dilate_roads(binary, dilation_kernel_size, dilation_iterations)

    # Step 6 (optional): Skeletonize
    if skeletonize_output:
        binary = extract_skeleton(binary, dilate_width=dilation_kernel_size)

    return binary


def postprocess_from_cfg(prob_map: np.ndarray, threshold: float, cfg: dict) -> np.ndarray:
    """Apply postprocessing using values from config dict."""
    pp = cfg.get("postprocessing", {})
    if not pp.get("apply", True):
        return (prob_map > threshold).astype(np.uint8) * 255

    return postprocess(
        prob_map=prob_map,
        threshold=threshold,
        blur_kernel=pp.get("blur_kernel", 3),
        closing_kernel_size=pp.get("closing_kernel_size", 7),
        closing_iterations=pp.get("closing_iterations", 2),
        dilation_kernel_size=pp.get("dilation_kernel_size", 3),
        dilation_iterations=pp.get("dilation_iterations", 1),
        min_component_size=pp.get("min_component_size", 500),
        skeletonize_output=pp.get("skeletonize", False),
    )
