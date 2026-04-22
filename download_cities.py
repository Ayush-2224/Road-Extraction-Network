"""
download_cities.py  (parallelised edition)
==========================================
Downloads satellite imagery + OSM roads for 5 Indian cities.
Tiles are fetched in parallel (5 threads) → ~5x faster than sequential.

Output:
  data/maps/   → <city>.tif   (GeoTIFF EPSG:3857, LZW)
  data/masks/  → <city>.geojson  (OSM roads WGS-84)
"""

import io, json, os, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import mercantile
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from PIL import Image
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────
# 5 medium Indian cities
# ─────────────────────────────────────────────────────────────────
CITIES = [
    ("bhopal",   77.3900, 23.2100, 77.4500, 23.2600),
    ("jabalpur", 79.9400, 23.1400, 80.0000, 23.1900),
    ("gwalior",  78.1500, 26.1900, 78.2100, 26.2400),
    ("ujjain",   75.7600, 23.1650, 75.8200, 23.2150),
    ("nagpur",   79.0600, 21.1200, 79.1200, 21.1700),
]

ZOOM          = 18
MAP_DIR       = "data/maps"
MASK_DIR      = "data/masks"
TILE_WORKERS  = 16      # parallel tile download threads (safe max ~24 before Google throttles)
MAX_RETRIES   = 3

OVERPASS_ENDPOINTS = [
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

Image.MAX_IMAGE_PIXELS = None
os.makedirs(MAP_DIR,  exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# Tile downloader (called from thread pool)
# ─────────────────────────────────────────────────────────────────
def _fetch_tile(tile, min_x, min_y):
    """Download one tile. Returns (paste_x, paste_y, PIL.Image or None)."""
    url = (f"http://mt{tile.x % 4}.google.com/vt/lyrs=s&hl=en"
           f"&x={tile.x}&y={tile.y}&z={tile.z}")
    px = (tile.x - min_x) * 256
    py = (tile.y - min_y) * 256

    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                img = Image.open(io.BytesIO(resp.read()))
                img.load()          # force decode inside the thread
            return (px, py, img)
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return (px, py, None)           # failed after retries


# ─────────────────────────────────────────────────────────────────
# Satellite download  (parallel tiles)
# ─────────────────────────────────────────────────────────────────
def download_satellite(name, west, south, east, north):
    out_path = os.path.join(MAP_DIR, f"{name}.tif")
    if os.path.exists(out_path):
        print(f"  [SKIP] {out_path} already exists")
        return out_path

    tiles = list(mercantile.tiles(west, south, east, north, ZOOM))
    min_x  = min(t.x for t in tiles);  max_x = max(t.x for t in tiles)
    min_y  = min(t.y for t in tiles);  max_y = max(t.y for t in tiles)
    W_px   = (max_x - min_x + 1) * 256
    H_px   = (max_y - min_y + 1) * 256
    canvas = Image.new("RGB", (W_px, H_px))

    print(f"  {len(tiles)} tiles  ({W_px}×{H_px} px)  "
          f"| {TILE_WORKERS} threads ...")

    failed = 0
    with ThreadPoolExecutor(max_workers=TILE_WORKERS) as pool:
        futures = {pool.submit(_fetch_tile, t, min_x, min_y): t for t in tiles}
        pbar    = tqdm(total=len(tiles), desc=f"  {name:10s}",
                       unit="tile", dynamic_ncols=True, leave=False)

        for fut in as_completed(futures):
            px, py, img = fut.result()
            if img:
                canvas.paste(img, (px, py))
            else:
                failed += 1
            pbar.update(1)

        pbar.close()

    if failed:
        print(f"  [WARN] {failed} tiles failed (blank patches)")

    # Georeference → EPSG:3857
    tl = mercantile.xy_bounds(mercantile.Tile(min_x, min_y, ZOOM))
    br = mercantile.xy_bounds(mercantile.Tile(max_x, max_y, ZOOM))
    transform = from_bounds(tl.left, br.bottom, br.right, tl.top, W_px, H_px)

    arr = np.moveaxis(np.array(canvas), -1, 0)   # (H,W,3)→(3,H,W)
    with rasterio.open(out_path, "w", driver="GTiff",
                       height=H_px, width=W_px, count=3,
                       dtype=arr.dtype, crs="EPSG:3857",
                       transform=transform, compress="lzw") as dst:
        dst.write(arr)

    mb = os.path.getsize(out_path) / 1024**2
    print(f"  [OK] {out_path}  ({mb:.1f} MB)")
    return out_path


# ─────────────────────────────────────────────────────────────────
# OSM road download via Overpass
# ─────────────────────────────────────────────────────────────────
def download_osm_roads(name, west, south, east, north):
    out_path = os.path.join(MASK_DIR, f"{name}.geojson")
    if os.path.exists(out_path):
        print(f"  [SKIP] {out_path} already exists")
        return out_path

    query = (f"[out:json][timeout:60];"
             f"(way[\"highway\"]({south},{west},{north},{east}););"
             f"(._;>;);out body;")
    post  = urllib.parse.urlencode({"data": query}).encode()

    result = None
    for ep in OVERPASS_ENDPOINTS:
        try:
            req = urllib.request.Request(ep, data=post)
            with urllib.request.urlopen(req, timeout=90) as r:
                result = json.loads(r.read().decode())
            break
        except Exception as e:
            print(f"  [WARN] {ep}: {e}")
            time.sleep(3)

    if not result:
        print(f"  [ERROR] Overpass failed for {name}")
        return None

    nodes = {el["id"]: (el["lon"], el["lat"])
             for el in result["elements"] if el["type"] == "node"}
    features = []
    for el in result["elements"]:
        if el["type"] != "way": continue
        coords = [nodes[n] for n in el.get("nodes", []) if n in nodes]
        if len(coords) < 2: continue
        features.append({
            "type": "Feature",
            "properties": {"id": el["id"], **el.get("tags", {})},
            "geometry": {"type": "LineString", "coordinates": coords}
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)

    print(f"  [OK] {out_path}  ({len(features)} road segments)")
    return out_path


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    print(f"Downloading {len(CITIES)} cities  |  {TILE_WORKERS} parallel threads\n")
    t0      = time.time()
    results = []

    for idx, (name, w, s, e, n) in enumerate(CITIES, 1):
        print(f"[{idx}/{len(CITIES)}] {name.upper()}")
        t1 = time.time()

        map_path  = download_satellite(name, w, s, e, n)
        mask_path = download_osm_roads(name, w, s, e, n)

        elapsed = time.time() - t1
        results.append({"name": name, "bbox": [w,s,e,n],
                        "map": map_path, "mask": mask_path})
        print(f"  Done in {elapsed:.0f}s\n")
        if idx < len(CITIES):
            time.sleep(3)   # brief pause between cities (Overpass rate limit)

    # Save manifest for training script
    manifest = {
        "zoom": ZOOM, "map_dir": MAP_DIR, "mask_dir": MASK_DIR,
        "cities": [r for r in results if r["map"] and r["mask"]]
    }
    os.makedirs("data", exist_ok=True)
    with open("data/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    total = (time.time() - t0) / 60
    print("=" * 50)
    print(f"DONE  ({total:.1f} min total)")
    for r in results:
        status = "OK" if (r["map"] and r["mask"]) else "FAILED"
        print(f"  {r['name']:12s} {status}")
    print(f"\n  Manifest → data/manifest.json")


if __name__ == "__main__":
    main()
