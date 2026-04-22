import mercantile
import requests
import io
from PIL import Image
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from tqdm import tqdm
import time

# Suppress PIL decompression bomb warning for large images
Image.MAX_IMAGE_PIXELS = None

def download_city_area(west, south, east, north, zoom=18, output_filename="indore/indore_map.tif"):
    print(f"Calculating tiles for zoom level {zoom}...")
    tiles = list(mercantile.tiles(west, south, east, north, zoom))
    print(f"Total tiles to download: {len(tiles)}")

    min_x, max_x = min(t.x for t in tiles), max(t.x for t in tiles)
    min_y, max_y = min(t.y for t in tiles), max(t.y for t in tiles)

    width_pixels  = (max_x - min_x + 1) * 256
    height_pixels = (max_y - min_y + 1) * 256
    print(f"Stitched image size: {width_pixels} x {height_pixels} px")

    stitched_image = Image.new('RGB', (width_pixels, height_pixels))
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })

    failed = 0
    for tile in tqdm(tiles, desc="Downloading Satellite Tiles"):
        url = (f"http://mt0.google.com/vt/lyrs=s&hl=en"
               f"&x={tile.x}&y={tile.y}&z={tile.z}")
        retries = 3
        success = False
        while retries > 0:
            try:
                response = session.get(url, timeout=15)
                if response.status_code == 200:
                    tile_img = Image.open(io.BytesIO(response.content))
                    paste_x = (tile.x - min_x) * 256
                    paste_y = (tile.y - min_y) * 256
                    stitched_image.paste(tile_img, (paste_x, paste_y))
                    success = True
                    break
            except Exception as e:
                pass
            retries -= 1
            time.sleep(1)
        if not success:
            failed += 1
        time.sleep(0.01)

    print(f"\nDownload complete. Failed tiles: {failed}/{len(tiles)}")
    print("Stitching and georeferencing image...")

    # Calculate geographic bounds in EPSG:3857 (Web Mercator)
    top_left_bounds    = mercantile.xy_bounds(mercantile.Tile(min_x, min_y, zoom))
    bottom_right_bounds = mercantile.xy_bounds(mercantile.Tile(max_x, max_y, zoom))

    transform = from_bounds(
        top_left_bounds.left,
        bottom_right_bounds.bottom,
        bottom_right_bounds.right,
        top_left_bounds.top,
        width_pixels,
        height_pixels
    )

    img_array = np.moveaxis(np.array(stitched_image), -1, 0)  # (H,W,3) -> (3,H,W)
    print(f"Image array shape: {img_array.shape}, dtype: {img_array.dtype}")
    print(f"Pixel value range: {img_array.min()} - {img_array.max()}")

    import os
    os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)

    with rasterio.open(
        output_filename,
        'w',
        driver='GTiff',
        height=height_pixels,
        width=width_pixels,
        count=3,
        dtype=img_array.dtype,
        crs='EPSG:3857',
        transform=transform,
        compress='lzw'
    ) as dest:
        dest.write(img_array)

    print(f"\n[OK] Georeferenced satellite image saved to: {output_filename}")
    print(f"     CRS: EPSG:3857 | Size: {width_pixels}x{height_pixels} px")


# Target Coordinates for Indore
w, s, e, n = 75.8300, 22.7000, 75.8900, 22.7500

download_city_area(w, s, e, n)
