"""
Inference + Evaluation Pipeline for Indore Road Extraction
==========================================================
- Loads runs/best_model.pth  (U-Net ResNet50, smp)
- Crops the target bbox from indore/indore_map.tif  (EPSG:3857)
- Runs sliding-window inference with 4x-TTA on GPU
- Applies morphological post-processing
- Saves predicted mask + overlay PNGs
- Builds ground-truth mask from indore/export.geojson (WGS-84 lon/lat)
- GT mask is pixel-aligned to match predicted mask exactly
- Computes IoU, F1, Precision, Recall and saves metrics.json
"""

import os, sys, json, time, warnings
warnings.filterwarnings("ignore")

# Force UTF-8 console output on Windows
if hasattr(sys.stdout, "fileno"):
    try:
        sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
        sys.stderr = open(sys.stderr.fileno(), mode="w", encoding="utf-8", buffering=1)
    except Exception:
        pass

import numpy as np
import cv2
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds as window_from_bounds
from pyproj import Transformer
from PIL import Image, ImageDraw

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
TIF_PATH     = "indore/indore_map.tif"
GEOJSON_PATH = "indore/real_export.geojson"
CHECKPOINT   = "runs/final_model_v3.pth"
OUTPUT_DIR   = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target bounding box in WGS-84 (lon/lat)
W_LON, S_LAT, E_LON, N_LAT = 75.8300, 22.7000, 75.8900, 22.7500

PATCH_SIZE   = 512
OVERLAP      = 0.5    # 50 % overlap
THRESHOLD    = 0.45   # optimal threshold from checkpoint
BATCH_SIZE   = 8      # patches per GPU batch (4x TTA inside)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Post-processing (from saved config)
CLOSING_KERNEL  = 7
CLOSING_ITERS   = 2
DILATION_KERNEL = 3
DILATION_ITERS  = 1
MIN_COMPONENT   = 500

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[INFO] GPU    : {torch.cuda.get_device_name(0)}")


# ─────────────────────────────────────────────────────────────────
# 1. LOAD MODEL
# ─────────────────────────────────────────────────────────────────
def load_model(path: str) -> nn.Module:
    print(f"\n[MODEL] Loading: {path}")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    state = None
    if isinstance(ckpt, dict):
        state = (ckpt.get("model_state_dict")
                 or ckpt.get("state_dict")
                 or ckpt.get("model"))
    state = state if state is not None else ckpt
    model = smp.Unet(
        encoder_name="resnet50", encoder_weights=None,
        in_channels=3, classes=1, activation=None,
    )
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    print("[MODEL] U-Net (ResNet50) loaded OK.")
    return model


# ─────────────────────────────────────────────────────────────────
# 2. CROP REGION FROM TIF  (handles any CRS internally)
# ─────────────────────────────────────────────────────────────────
def crop_tif(tif_path: str,
             w_lon: float, s_lat: float, e_lon: float, n_lat: float):
    """
    Crops the TIF to a WGS-84 bounding box.
    - Reprojects bbox to the TIF's native CRS before cropping.
    Returns: (H, W, 3) uint8 RGB array, pixel_transform (Affine in TIF CRS).
    """
    print(f"\n[TIF] Cropping  W={w_lon} S={s_lat} E={e_lon} N={n_lat}  (WGS-84)")
    with rasterio.open(tif_path) as src:
        tif_crs = src.crs
        print(f"[TIF] Native CRS : {tif_crs}")

        # Reproject bbox from EPSG:4326 → TIF CRS
        if tif_crs.to_epsg() != 4326:
            left, bottom, right, top = transform_bounds(
                CRS.from_epsg(4326), tif_crs,
                w_lon, s_lat, e_lon, n_lat
            )
        else:
            left, bottom, right, top = w_lon, s_lat, e_lon, n_lat

        print(f"[TIF] Bbox in TIF CRS: ({left:.2f}, {bottom:.2f}, {right:.2f}, {top:.2f})")
        window = window_from_bounds(left, bottom, right, top, src.transform)
        data   = src.read(window=window)           # (bands, H, W)
        pixel_transform = src.window_transform(window)

    data = np.clip(data, 0, 255).astype(np.uint8)
    img  = np.transpose(data[:3], (1, 2, 0))      # → (H, W, 3) RGB
    print(f"[TIF] Cropped : {img.shape[1]} x {img.shape[0]} px  "
          f"| pixel range [{img.min()}, {img.max()}]  mean={img.mean():.1f}")
    return img, pixel_transform


# ─────────────────────────────────────────────────────────────────
# 3. SLIDING-WINDOW INFERENCE WITH 4x TTA
# ─────────────────────────────────────────────────────────────────
def _preprocess(patch: np.ndarray) -> torch.Tensor:
    p = patch.astype(np.float32) / 255.0
    p = (p - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(p.transpose(2, 0, 1)).unsqueeze(0)


def _tta_batch(model: nn.Module, patches: list[np.ndarray]) -> list[np.ndarray]:
    """
    4-fold TTA on a list of patches.
    Returns list of (H, W) float32 probability maps.
    """
    flips = [
        (None,  None),   # original
        (None,  1),      # horizontal flip
        (0,     None),   # vertical flip
        (0,     1),      # both
    ]
    acc = [np.zeros(p.shape[:2], dtype=np.float32) for p in patches]

    for fy, fx in flips:
        tensors = []
        for p in patches:
            aug = p
            if fy is not None: aug = np.flip(aug, axis=fy).copy()
            if fx is not None: aug = np.flip(aug, axis=fx).copy()
            tensors.append(_preprocess(aug))

        batch = torch.cat(tensors, 0).to(DEVICE)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
            probs = torch.sigmoid(model(batch)).squeeze(1).cpu().numpy()  # (N,H,W)

        for i, prob in enumerate(probs):
            if fy is not None: prob = np.flip(prob, axis=0).copy()
            if fx is not None: prob = np.flip(prob, axis=1).copy()
            acc[i] += prob

    return [a / len(flips) for a in acc]


def run_inference(model: nn.Module, image: np.ndarray) -> np.ndarray:
    """Sliding-window + gaussian-blend inference. Returns (H, W) float32 prob map."""
    H, W = image.shape[:2]
    stride = int(PATCH_SIZE * (1 - OVERLAP))

    # Reflect-pad so the image fills full patch windows
    pad_h = (-H % stride) if H % stride else 0
    pad_w = (-W % stride) if W % stride else 0
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    pH, pW = padded.shape[:2]

    prob_map  = np.zeros((pH, pW), dtype=np.float32)
    count_map = np.zeros((pH, pW), dtype=np.float32)

    # Gaussian blend weight (center-heavy)
    g1d   = np.exp(-np.linspace(-3, 3, PATCH_SIZE) ** 2 / 2).astype(np.float32)
    gauss = np.outer(g1d, g1d)

    # Build all positions (ensure full coverage including edges)
    ys = list(range(0, pH - PATCH_SIZE + 1, stride))
    xs = list(range(0, pW - PATCH_SIZE + 1, stride))
    if ys[-1] + PATCH_SIZE < pH: ys.append(pH - PATCH_SIZE)
    if xs[-1] + PATCH_SIZE < pW: xs.append(pW - PATCH_SIZE)
    positions = list({(y, x) for y in ys for x in xs})

    total = len(positions)
    print(f"[INFER] Patches: {total}  |  Batch: {BATCH_SIZE}  |  TTA: 4x")

    for i in range(0, total, BATCH_SIZE):
        pos   = positions[i: i + BATCH_SIZE]
        crops = [padded[y: y + PATCH_SIZE, x: x + PATCH_SIZE] for y, x in pos]
        preds = _tta_batch(model, crops)
        for (y, x), pred in zip(pos, preds):
            prob_map [y: y + PATCH_SIZE, x: x + PATCH_SIZE] += pred  * gauss
            count_map[y: y + PATCH_SIZE, x: x + PATCH_SIZE] += gauss
        done = min(i + BATCH_SIZE, total)
        print(f"\r[INFER] {done}/{total}  ({100*done/total:.1f}%)", end="", flush=True)

    print()
    prob_map /= np.maximum(count_map, 1e-8)
    return prob_map[:H, :W]


# ─────────────────────────────────────────────────────────────────
# 4. POST-PROCESSING
# ─────────────────────────────────────────────────────────────────
def postprocess(prob: np.ndarray, thr: float = THRESHOLD) -> np.ndarray:
    """Threshold → close → remove small blobs → dilate. Returns uint8 {0,255}."""
    binary = (prob >= thr).astype(np.uint8)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSING_KERNEL,)*2)
    for _ in range(CLOSING_ITERS):
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean = np.zeros_like(binary)
    for c in range(1, n):
        if stats[c, cv2.CC_STAT_AREA] >= MIN_COMPONENT:
            clean[lbl == c] = 1

    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATION_KERNEL,)*2)
    for _ in range(DILATION_ITERS):
        clean = cv2.dilate(clean, k_dil)

    return (clean * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────
# 5. GROUND-TRUTH MASK FROM GEOJSON  (WGS-84 lon/lat LineStrings)
# ─────────────────────────────────────────────────────────────────
def geojson_to_mask(geojson_path: str,
                    w_lon: float, s_lat: float, e_lon: float, n_lat: float,
                    target_h: int, target_w: int,
                    road_width_px: int = 15) -> np.ndarray:
    """
    Rasterizes road LineStrings from GeoJSON to a (target_h, target_w) uint8 mask.
    GeoJSON coords are in WGS-84 lon/lat; bbox defines the pixel viewport.
    """
    print(f"\n[GT] Rasterizing GeoJSON: {geojson_path}")
    with open(geojson_path, encoding="utf-8") as f:
        gj = json.load(f)

    lon_span = e_lon - w_lon
    lat_span = n_lat - s_lat

    def to_px(lon, lat):
        px = int((lon - w_lon) / lon_span * target_w)
        py = int((n_lat - lat) / lat_span * target_h)   # y-axis inverted
        return px, py

    mask_img = Image.new("L", (target_w, target_h), 0)
    draw     = ImageDraw.Draw(mask_img)
    buf      = 0.001
    drawn    = 0

    for feat in gj["features"]:
        geom = feat["geometry"]
        gtype = geom["type"]
        if gtype == "LineString":
            segs = [geom["coordinates"]]
        elif gtype == "MultiLineString":
            segs = geom["coordinates"]
        else:
            continue

        for coords in segs:
            in_box = any(
                (w_lon - buf) <= lon <= (e_lon + buf) and
                (s_lat - buf) <= lat <= (n_lat + buf)
                for lon, lat, *_ in coords
            )
            if not in_box:
                continue
            pts = [to_px(lon, lat) for lon, lat, *_ in coords]
            if len(pts) >= 2:
                draw.line(pts, fill=255, width=road_width_px)
                drawn += 1

    print(f"[GT] Drew {drawn} road segments  →  "
          f"{target_w}x{target_h} px mask")
    return np.array(mask_img, dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────
# 6. METRICS
# ─────────────────────────────────────────────────────────────────
def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    p = (pred > 127).astype(np.float32)
    g = (gt   > 127).astype(np.float32)
    tp = float(np.sum(p * g))
    fp = float(np.sum(p * (1 - g)))
    fn = float(np.sum((1 - p) * g))
    tn = float(np.sum((1 - p) * (1 - g)))
    return {
        "IoU"      : tp / (tp + fp + fn + 1e-8),
        "F1"       : 2*tp / (2*tp + fp + fn + 1e-8),
        "Precision": tp / (tp + fp + 1e-8),
        "Recall"   : tp / (tp + fn + 1e-8),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


# ─────────────────────────────────────────────────────────────────
# 7. VISUALISATION HELPERS
# ─────────────────────────────────────────────────────────────────
def overlay(img: np.ndarray, mask: np.ndarray,
            color=(0, 120, 255), alpha=0.5) -> np.ndarray:
    out = img.copy()
    px  = mask > 127
    out[px] = ((1 - alpha) * out[px] + alpha * np.array(color)).astype(np.uint8)
    return out


def save_comparison(img, pred, gt, path):
    H = img.shape[0]
    panel_pred = overlay(img, pred, color=(0, 120, 255))
    panel_gt   = overlay(img, gt,   color=(255, 220, 0))

    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.8, H / 2500)
    thick = max(2, int(H / 600))
    for panel, text in [(panel_pred, "Predicted"), (panel_gt, "Ground Truth")]:
        bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, text, (30, 80), font, scale*2, (255,255,255), thick+2, cv2.LINE_AA)
        cv2.putText(bgr, text, (30, 80), font, scale*2, (20, 20, 20),  thick,   cv2.LINE_AA)
        # copy back
        panel[:] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    combined = np.concatenate([
        cv2.cvtColor(img,        cv2.COLOR_RGB2BGR),
        cv2.cvtColor(panel_pred, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(panel_gt,   cv2.COLOR_RGB2BGR),
    ], axis=1)

    if combined.shape[1] > 6000:
        s = 6000 / combined.shape[1]
        combined = cv2.resize(combined, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, combined)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    # 1. Model
    model = load_model(CHECKPOINT)

    # 2. Satellite image crop
    image, pix_transform = crop_tif(TIF_PATH, W_LON, S_LAT, E_LON, N_LAT)
    H, W_img = image.shape[:2]
    print(f"[INFO] Image : {W_img} x {H} px")

    # 3. Inference
    print("\n[INFER] Sliding-window inference on GPU ...")
    prob_map = run_inference(model, image)
    print(f"[INFER] prob_map: min={prob_map.min():.4f}  max={prob_map.max():.4f}  "
          f"mean={prob_map.mean():.4f}")

    # Save raw prob map (float32 rescaled to uint8)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "prob_map.png"),
                (prob_map * 255).clip(0, 255).astype(np.uint8))

    # 4. Post-process
    pred_mask = postprocess(prob_map)
    road_frac = (pred_mask > 127).mean() * 100
    print(f"[POST] Predicted road coverage : {road_frac:.2f}% of image")

    cv2.imwrite(os.path.join(OUTPUT_DIR, "predicted_mask.png"), pred_mask)
    print(f"[SAVE] predicted_mask.png")

    cv2.imwrite(os.path.join(OUTPUT_DIR, "predicted_overlay.png"),
                cv2.cvtColor(overlay(image, pred_mask), cv2.COLOR_RGB2BGR))
    print(f"[SAVE] predicted_overlay.png")

    # 5. Ground-truth mask
    gt_raw = geojson_to_mask(GEOJSON_PATH, W_LON, S_LAT, E_LON, N_LAT, H, W_img)

    # 6. Align sizes (robustly resize GT to match pred if any border mismatch)
    pred_h, pred_w = pred_mask.shape[:2]
    gt_h,   gt_w   = gt_raw.shape[:2]
    if (gt_h, gt_w) != (pred_h, pred_w):
        print(f"[ALIGN] GT {gt_w}x{gt_h}  -->  pred {pred_w}x{pred_h}  (resizing GT)")
        gt_mask = cv2.resize(gt_raw, (pred_w, pred_h), interpolation=cv2.INTER_NEAREST)
    else:
        gt_mask = gt_raw

    gt_road_frac = (gt_mask > 127).mean() * 100
    print(f"[GT] Ground-truth road coverage : {gt_road_frac:.2f}% of image")

    cv2.imwrite(os.path.join(OUTPUT_DIR, "gt_mask.png"), gt_mask)
    print(f"[SAVE] gt_mask.png")

    cv2.imwrite(os.path.join(OUTPUT_DIR, "gt_overlay.png"),
                cv2.cvtColor(overlay(image, gt_mask, color=(255, 220, 0)), cv2.COLOR_RGB2BGR))
    print(f"[SAVE] gt_overlay.png")

    # 7. Comparison panel
    save_comparison(image, pred_mask, gt_mask,
                    os.path.join(OUTPUT_DIR, "comparison.png"))
    print(f"[SAVE] comparison.png")

    # 8. Metrics
    print("\n[EVAL] Computing metrics ...")
    m = compute_metrics(pred_mask, gt_mask)

    sep = "=" * 52
    print(f"\n{sep}")
    print("  EVALUATION RESULTS")
    print(sep)
    print(f"  IoU       : {m['IoU']:.4f}   ({m['IoU']*100:.2f}%)")
    print(f"  F1 Score  : {m['F1']:.4f}   ({m['F1']*100:.2f}%)")
    print(f"  Precision : {m['Precision']:.4f}")
    print(f"  Recall    : {m['Recall']:.4f}")
    print(f"  TP={m['TP']:,}  FP={m['FP']:,}  FN={m['FN']:,}  TN={m['TN']:,}")
    print(sep)

    result = {
        "bbox"      : {"west": W_LON, "south": S_LAT, "east": E_LON, "north": N_LAT},
        "image_size": {"width": W_img, "height": H},
        "threshold" : THRESHOLD,
        "metrics"   : m,
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[SAVE] metrics.json")

    print(f"\n[DONE] Total time : {time.time()-t0:.1f}s")
    print(f"[INFO] Results    : {os.path.abspath(OUTPUT_DIR)}/")


if __name__ == "__main__":
    main()
