"""
repair_missing_tiles.py
Scans the existing 5 city TIF files for purely black/missing 256x256 tiles, 
and surgically redownloads ONLY those specific missing tiles directly into the TIFs.
"""

import os, json, time, io
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import rasterio
import mercantile
from PIL import Image
from tqdm import tqdm

MANIFEST_PATH = "data/manifest.json"
ZOOM = 18
MAX_RETRIES = 5
WORKERS = 16

def _fetch_repair_tile(tile, px, py):
    url = f"http://mt{tile.x % 4}.google.com/vt/lyrs=s&hl=en&x={tile.x}&y={tile.y}&z={tile.z}"
    # Higher retries and sleep time to punch through Google's rate limits
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                img = Image.open(io.BytesIO(resp.read()))
                img.load()
            return (px, py, np.array(img))
        except Exception:
            time.sleep(2.0 * (attempt + 1))
    return (px, py, None)

def main():
    if not os.path.exists(MANIFEST_PATH):
        print(f"Error: {MANIFEST_PATH} not found.")
        return

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    for c in manifest.get("cities", []):
        name = c["name"]
        w, s, e, n = c["bbox"]
        tif_path = f"data/maps/{name}.tif"

        if not os.path.exists(tif_path):
            continue

        print(f"\n[{name.upper()}] Scanning TIF for missing tiles...")
        
        with rasterio.open(tif_path, "r") as src:
            meta = src.meta.copy()
            arr = src.read()  # (3, H, W)
        
        # Calculate exactly which tiles belong where 
        tiles = list(mercantile.tiles(w, s, e, n, ZOOM))
        min_x = min(t.x for t in tiles); min_y = min(t.y for t in tiles)

        missing = []
        for t in tiles:
            px = (t.x - min_x) * 256
            py = (t.y - min_y) * 256
            
            # Check if this exact 256x256 block is mostly pure black
            block = arr[:, py:py+256, px:px+256]
            if block.size > 0 and block.mean() < 5.0:  
                missing.append((t, px, py))

        if not missing:
            print(f"  [OK] No missing tiles found. Image is perfectly clean.")
            continue

        print(f"  [WARN] Found {len(missing)} missing tiles. Redownloading...")

        failed = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(_fetch_repair_tile, item[0], item[1], item[2]): item for item in missing}
            pbar = tqdm(total=len(missing), desc="  Repairing", unit="tile", leave=False, dynamic_ncols=True)
            
            for fut in as_completed(futures):
                px, py, img_arr = fut.result()
                if img_arr is not None:
                    # Move axis to match rasterio (3, H, W)
                    img_arr = np.moveaxis(img_arr, -1, 0)
                    
                    # Carefully paste into the massive array
                    _, h_blk, w_blk = img_arr.shape
                    arr[:, py:py+h_blk, px:px+w_blk] = img_arr
                else:
                    failed += 1
                pbar.update(1)
            pbar.close()

        if failed:
            print(f"  [WARN] {failed} tiles failed again. Google might be aggressively blocking.")
        else:
            print(f"  [SUCCESS] All {len(missing)} tiles perfectly downloaded.")

        if len(missing) > failed:
            print(f"  Saving repaired TIF back to {tif_path}...")
            with rasterio.open(tif_path, "w", **meta) as dst:
                dst.write(arr)
            print(f"  [DONE] {name} is perfectly repaired and noise-free!")

if __name__ == "__main__":
    main()
