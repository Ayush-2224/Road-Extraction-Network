"""
finetune_multicity.py
=====================
Multi-city fine-tuning of U-Net (ResNet50) on Indian satellite imagery.

PHASE 1 - Patch extraction (one-time, cached to disk):
  For each city: crop TIF → rasterize GeoJSON → extract 512×512 patches
  Saves image/mask PNG pairs to data/patches/ (skips if already done)

PHASE 2 - Training:
  Loads patches from disk → augment → train with AMP on GPU
  Stops when val IoU >= TARGET_IOU (0.65)
  Best weights → runs/best_model_multicity.pth (also overwrites best_model.pth)

Cities used:
  • all cities in data/manifest.json  (bhopal, jabalpur, gwalior, ujjain, nagpur)
"""

import os, sys, json, time, random, glob
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds as win_from_bounds
from PIL import Image, ImageDraw
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
CHECKPOINT_IN   = "runs/checkpoints/best_model.pth"
CHECKPOINT_OUT  = "runs/final_model_v3.pth"
PATCH_DIR       = "data/patches"
MANIFEST_PATH   = "data/manifest.json"


PATCH_SIZE         = 512
STRIDE             = 256          # 50% overlap
ROAD_WIDTH_PX      = 15           # ~9 m at zoom-18 resolution
MAX_PER_CITY       = 2500         # Use massive dataset to stop overfitting

BATCH_SIZE         = 8
NUM_WORKERS        = 4            # 4 background workers for MUCH faster data loading
INIT_LR            = 1e-4         # 3x higher learning rate to converge faster
WEIGHT_DECAY       = 1e-3         # 10x higher L2 penalty to stop memorization
MAX_EPOCHS         = 30
TARGET_IOU         = 0.65
EARLY_STOP_PAT     = 12
VAL_SPLIT          = 0.15
THRESHOLD          = 0.45

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
os.makedirs(PATCH_DIR + "/images", exist_ok=True)
os.makedirs(PATCH_DIR + "/masks",  exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# PHASE 1 HELPERS
# ─────────────────────────────────────────────────────────────────
def load_tif_crop(tif_path, bbox):
    """Crop TIF to WGS-84 bbox. Handles any native CRS via reprojection."""
    w, s, e, n = bbox
    with rasterio.open(tif_path) as src:
        if src.crs.to_epsg() != 4326:
            l, b, r, t = transform_bounds(CRS.from_epsg(4326), src.crs, w, s, e, n)
        else:
            l, b, r, t = w, s, e, n
        window = win_from_bounds(l, b, r, t, src.transform)
        data   = src.read(window=window)
    img = np.transpose(np.clip(data[:3], 0, 255).astype(np.uint8), (1, 2, 0))
    return img  # (H, W, 3)


def rasterize_geojson(geojson_path, bbox, target_h, target_w,
                      road_width=ROAD_WIDTH_PX):
    """GeoJSON LineStrings → binary uint8 mask (0/255)."""
    w, s, e, n = bbox
    lon_span = e - w;  lat_span = n - s
    def to_px(lon, lat):
        return (int((lon - w) / lon_span * target_w),
                int((n - lat) / lat_span * target_h))

    with open(geojson_path, encoding="utf-8") as f:
        gj = json.load(f)

    canvas = Image.new("L", (target_w, target_h), 0)
    draw   = ImageDraw.Draw(canvas)
    buf    = 0.001

    for feat in gj["features"]:
        geom = feat["geometry"]
        segs = ([geom["coordinates"]] if geom["type"] == "LineString"
                else geom["coordinates"] if geom["type"] == "MultiLineString"
                else [])
        for coords in segs:
            if not any((w-buf) <= lo <= (e+buf) and (s-buf) <= la <= (n+buf)
                       for lo, la, *_ in coords):
                continue
            pts = [to_px(lo, la) for lo, la, *_ in coords]
            if len(pts) >= 2:
                draw.line(pts, fill=255, width=road_width)

    return np.array(canvas, dtype=np.uint8)


def is_blank(patch: np.ndarray) -> bool:
    """True if patch is mostly black/empty (failed tile download or border)."""
    return patch.mean() < 15 or patch.max() < 30


def extract_patches(image, mask, city_name, stride=STRIDE):
    """Extract, balance, cap, and save 512×512 patches to disk."""
    # Check how many already saved
    existing = glob.glob(f"{PATCH_DIR}/images/{city_name}_*.png")
    if existing:
        print(f"  [SKIP] {city_name}: {len(existing)} patches already saved")
        return len(existing)

    H, W = image.shape[:2]
    ys = list(range(0, H - PATCH_SIZE + 1, stride))
    xs = list(range(0, W - PATCH_SIZE + 1, stride))
    if ys[-1] + PATCH_SIZE < H: ys.append(H - PATCH_SIZE)
    if xs[-1] + PATCH_SIZE < W: xs.append(W - PATCH_SIZE)

    pos_patches, neg_patches = [], []
    skipped = 0
    for y in ys:
        for x in xs:
            ip = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE].copy()
            mp = mask [y:y+PATCH_SIZE, x:x+PATCH_SIZE].copy()
            if is_blank(ip):          # skip failed-tile / border black patches
                skipped += 1
                continue
            (pos_patches if (mp > 127).any() else neg_patches).append((ip, mp))
    if skipped:
        print(f"  [BLANK] {city_name}: skipped {skipped} blank patches")

    # Balance 1:1, cap at MAX_PER_CITY
    random.shuffle(pos_patches); random.shuffle(neg_patches)
    n_pos  = min(len(pos_patches), MAX_PER_CITY // 2)
    n_neg  = min(len(neg_patches), n_pos)
    chosen = pos_patches[:n_pos] + neg_patches[:n_neg]
    random.shuffle(chosen)

    # Save to disk as PNG
    for idx, (ip, mp) in enumerate(chosen):
        img_name  = f"{PATCH_DIR}/images/{city_name}_{idx:05d}.png"
        mask_name = f"{PATCH_DIR}/masks/{city_name}_{idx:05d}.png"
        cv2.imwrite(img_name,  cv2.cvtColor(ip, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_name, mp)

    road_cov = np.mean([(m > 127).mean() for _, m in chosen]) * 100
    print(f"  {city_name:12s} | {len(chosen):>4} patches "
          f"(pos={n_pos} neg={n_neg}) | road coverage ~{road_cov:.1f}%")
    return len(chosen)


# ─────────────────────────────────────────────────────────────────
# PHASE 1 MAIN — build patch cache
# ─────────────────────────────────────────────────────────────────
def prepare_all_patches(cities):
    print("\n" + "=" * 60)
    print("PHASE 1 — Patch Extraction")
    print("=" * 60)
    total = 0
    for c in cities:
        name    = c["name"]
        tif     = c["tif"]
        geojson = c["geojson"]
        bbox    = c["bbox"]
        print(f"\n[{name.upper()}]  {tif}")

        if not os.path.exists(tif):
            print(f"  [SKIP] TIF not found: {tif}")
            continue
        if not os.path.exists(geojson):
            print(f"  [SKIP] GeoJSON not found: {geojson}")
            continue

        image = load_tif_crop(tif, bbox)
        H, W  = image.shape[:2]
        print(f"  Image: {W}×{H} px  | mean pixel={image.mean():.1f}")

        mask  = rasterize_geojson(geojson, bbox, H, W)
        cov   = (mask > 127).mean() * 100
        print(f"  GT mask: road coverage={cov:.2f}%")

        n = extract_patches(image, mask, name)
        total += n

    all_imgs  = sorted(glob.glob(f"{PATCH_DIR}/images/*.png"))
    all_masks = sorted(glob.glob(f"{PATCH_DIR}/masks/*.png"))
    print(f"\n[PHASE 1 DONE] Total patches: {len(all_imgs)}")
    return all_imgs, all_masks


# ─────────────────────────────────────────────────────────────────
# PHASE 2 — Dataset / Augmentations
# ─────────────────────────────────────────────────────────────────
train_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25,
                         val_shift_limit=15, p=0.4),
    A.GaussNoise(var_limit=(10.0, 40.0), p=0.3),
    A.Blur(blur_limit=3, p=0.2),
    A.ElasticTransform(alpha=80, sigma=6, p=0.3),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

val_aug = A.Compose([
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


class PatchDataset(Dataset):
    def __init__(self, img_paths, mask_paths, aug=None):
        self.imgs  = img_paths
        self.masks = mask_paths
        self.aug   = aug

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img  = cv2.cvtColor(cv2.imread(self.imgs[idx]),  cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)

        # Safety: drop blank patches that slipped through extraction
        if img is None or is_blank(img):
            return None

        bin_mask = (mask > 127).astype(np.uint8)

        if self.aug:
            out  = self.aug(image=img, mask=bin_mask)
            img  = out["image"]
            mask = out["mask"].float().unsqueeze(0)
        else:
            img  = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255.0)
            mask = torch.from_numpy(bin_mask).float().unsqueeze(0)
        return img, mask


def collate_skip_none(batch):
    """Drop None samples (blank patches) from a batch."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    imgs  = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    return imgs, masks


def make_split(img_paths, mask_paths):
    pairs   = list(zip(img_paths, mask_paths))
    random.shuffle(pairs)
    n_val   = max(1, int(len(pairs) * VAL_SPLIT))
    val_p   = pairs[:n_val]
    train_p = pairs[n_val:]
    ti, tm  = zip(*train_p); vi, vm = zip(*val_p)
    print(f"[SPLIT] Train: {len(train_p)}  |  Val: {len(val_p)}")
    return list(ti), list(tm), list(vi), list(vm)


# ─────────────────────────────────────────────────────────────────
# PHASE 2 — Loss / Metric
# ─────────────────────────────────────────────────────────────────
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce  = self.bce(logits, targets)
        prob = torch.sigmoid(logits)
        tp   = (prob * targets).sum(dim=(1,2,3))
        fp   = (prob * (1 - targets)).sum(dim=(1,2,3))
        fn   = ((1 - prob) * targets).sum(dim=(1,2,3))
        dice = 1 - (2*tp + 1) / (2*tp + fp + fn + 1)
        return 0.5 * bce + 0.5 * dice.mean()


@torch.no_grad()
def iou_score(logits, targets, thr=THRESHOLD):
    pred = (torch.sigmoid(logits) > thr).float()
    tp   = (pred * targets).sum(dim=(1,2,3))
    fp   = (pred * (1 - targets)).sum(dim=(1,2,3))
    fn   = ((1 - pred) * targets).sum(dim=(1,2,3))
    return (tp / (tp + fp + fn + 1e-6)).mean().item()


# ─────────────────────────────────────────────────────────────────
# PHASE 2 — Model
# ─────────────────────────────────────────────────────────────────
def load_model():
    print(f"\n[MODEL] Loading: {CHECKPOINT_IN}")
    ckpt  = torch.load(CHECKPOINT_IN, map_location=DEVICE, weights_only=False)
    state = (ckpt.get("model_state_dict") or ckpt.get("state_dict")
             or ckpt.get("model")) if isinstance(ckpt, dict) else ckpt
    model = smp.Unet(encoder_name="resnet50", encoder_weights=None,
                     in_channels=3, classes=1, activation=None)
    model.load_state_dict(state if state is not None else ckpt)
    model.to(DEVICE)
    print("[MODEL] U-Net (ResNet50) loaded OK.")
    return model


# ─────────────────────────────────────────────────────────────────
# PHASE 2 — Training loop
# ─────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, scaler,
              train: bool, epoch: int):
    model.train() if train else model.eval()
    total_loss = total_iou = n = 0
    phase = "Train" if train else "Val  "
    ctx   = torch.enable_grad() if train else torch.no_grad()

    bar = tqdm(loader, desc=f"  Ep {epoch:>3}/{MAX_EPOCHS} [{phase}]",
               unit="batch", dynamic_ncols=True, leave=False)

    with ctx:
        for batch in bar:
            if batch is None:   # entire batch was blank
                continue
            imgs, masks = batch
            imgs  = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss   = criterion(logits, masks)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()

            total_loss += loss.item()
            total_iou  += iou_score(logits.detach(), masks)
            n          += 1
            bar.set_postfix(loss=f"{total_loss/n:.4f}",
                            iou =f"{total_iou/n:.4f}")

    bar.close()
    return total_loss / n, total_iou / n


def train(model, train_ds, val_ds):
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              collate_fn=collate_skip_none)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              collate_fn=collate_skip_none)

    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=INIT_LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=INIT_LR * 0.01)
    scaler    = torch.cuda.amp.GradScaler()

    best_iou = 0.0
    patience = 0
    log      = []

    sep = "=" * 76
    print(f"\n{sep}")
    print(f"  {'Ep':>3}  {'TrLoss':>8}  {'TrIoU':>7}  "
          f"{'VaLoss':>8}  {'VaIoU':>7}  {'LR':>9}  {'Time':>6}  Notes")
    print(sep)

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_iou = run_epoch(model, train_loader, criterion,
                                    optimizer, scaler, True,  epoch)
        vl_loss, vl_iou = run_epoch(model, val_loader,   criterion,
                                    optimizer, scaler, False, epoch)
        scheduler.step()
        lr      = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        is_best = vl_iou > best_iou

        if is_best:
            best_iou = vl_iou
            patience = 0
            state = {"model_state_dict": model.state_dict(),
                     "epoch": epoch, "val_iou": best_iou,
                     "threshold": THRESHOLD}
            torch.save(state, CHECKPOINT_OUT)
            torch.save(state, "runs/best_model.pth")   # overwrite so run_inference picks it up
        else:
            patience += 1

        note = "<-- BEST" if is_best else (
               f"patience {patience}/{EARLY_STOP_PAT}" if patience > 3 else "")
        print(f"  {epoch:>3}  {tr_loss:>8.4f}  {tr_iou:>7.4f}  "
              f"{vl_loss:>8.4f}  {vl_iou:>7.4f}  {lr:>9.2e}  "
              f"{elapsed:>5.0f}s  {note}")

        log.append({"epoch": epoch, "train_loss": tr_loss, "train_iou": tr_iou,
                    "val_loss": vl_loss, "val_iou": vl_iou,
                    "lr": lr, "time_s": round(elapsed, 1)})

        if best_iou >= TARGET_IOU:
            print(f"\n[TARGET] Val IoU {best_iou:.4f} >= {TARGET_IOU} -- stopping!")
            break
        if patience >= EARLY_STOP_PAT:
            print(f"\n[STOP] No improvement for {EARLY_STOP_PAT} epochs. "
                  f"Best={best_iou:.4f}")
            break

    print(sep)
    print(f"\n[RESULT] Best Val IoU : {best_iou:.4f}  ({best_iou*100:.2f}%)")
    print(f"[SAVED]  {CHECKPOINT_OUT}")
    print(f"[SAVED]  runs/best_model.pth  (overwritten for inference)")

    with open("runs/finetune_multicity_log.json", "w") as f:
        json.dump(log, f, indent=2)
    return best_iou


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    print(f"\n[INFO] Device : {DEVICE}  |  {torch.cuda.get_device_name(0)}")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    t_start = time.time()

    # Build city list: Indore + manifest cities
    cities = []
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
        for c in manifest.get("cities", []):
            name    = c["name"]
            w, s, e, n = c["bbox"]
            cities.append({
                "name"    : name,
                "tif"     : f"data/maps/{name}.tif",
                "geojson" : f"data/masks/{name}.geojson",
                "bbox"    : [w, s, e, n],
            })
    else:
        print(f"[WARN] {MANIFEST_PATH} not found — no cities to train on")
        return
        

    # Phase 1: extract patches
    img_paths, mask_paths = prepare_all_patches(cities)
    if not img_paths:
        print("[ERROR] No patches found. Check TIF/GeoJSON paths.")
        return

    # Phase 2: train
    ti, tm, vi, vm = make_split(img_paths, mask_paths)
    train_ds = PatchDataset(ti, tm, aug=train_aug)
    val_ds   = PatchDataset(vi, vm, aug=val_aug)
    print(f"[DS] Train batches: {len(train_ds)//BATCH_SIZE}  |  "
          f"Val batches: {len(val_ds)//BATCH_SIZE}")

    model    = load_model()
    best_iou = train(model, train_ds, val_ds)

    wall = (time.time() - t_start) / 60
    print(f"\n[TOTAL] Wall time: {wall:.1f} min")
    if best_iou >= TARGET_IOU:
        print("Run `python run_inference.py` to evaluate on the full Indore map.")


if __name__ == "__main__":
    main()
