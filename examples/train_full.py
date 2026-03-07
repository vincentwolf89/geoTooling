"""Opgeschaalde training: alle trajecten, DTM + luchtfoto.

Downloadt AHN4 DTM + PDOK luchtfoto voor meerdere secties van alle
trajecten, genereert labels, knipt tiles, en traint een U-Net met
5 kanalen (DTM, slope, R, G, B).

Gebruik:
    python examples/train_full.py
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from io import BytesIO

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import LineString, shape
import requests
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rasterio.plot import show

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kruinlijn.pipeline import kniklijnen_pipeline, generate_training_labels

# --- Configuratie ---
AHN4_URL = (
    "https://ahn.arcgisonline.nl/arcgis/rest/services"
    "/Hoogtebestand/AHN4_DTM_50cm/ImageServer"
)
LUCHTFOTO_URL = (
    "https://services.arcgisonline.nl/arcgis/rest/services"
    "/Luchtfoto/Luchtfoto/MapServer/export"
)

OUTPUT_DIR = Path("output/dl_full")
GEOJSON_PATH = Path("data/raw/Trajecten ZWO.geojson")

# Per traject: secties van elk SECTION_LENGTH_M lang
SECTION_LENGTH_M = 2000
SECTION_SPACING_M = 2500  # afstand tussen secties (voor overlap/diversiteit)
BUFFER_M = 50
TILE_SIZE = 256
TILE_OVERLAP = 64
EPOCHS = 50


def download_ahn4_dtm(bbox: tuple, output_path: Path, size_px: int = 2048) -> None:
    """Download AHN4 DTM via ArcGIS ImageServer."""
    minx, miny, maxx, maxy = bbox
    width = maxx - minx
    height = maxy - miny
    aspect = width / height

    if aspect >= 1:
        px_w = size_px
        px_h = max(64, int(size_px / aspect))
    else:
        px_h = size_px
        px_w = max(64, int(size_px * aspect))

    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "bboxSR": "28992", "imageSR": "28992",
        "size": f"{px_w},{px_h}",
        "format": "tiff", "pixelType": "F32",
        "noData": "-9999",
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }

    resp = requests.get(f"{AHN4_URL}/exportImage", params=params, timeout=120)
    resp.raise_for_status()

    if resp.headers.get("content-type", "").startswith("application/json"):
        raise RuntimeError(f"AHN4 error: {resp.json()}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_suffix(".raw.tif")
    raw_path.write_bytes(resp.content)

    transform = from_bounds(minx, miny, maxx, maxy, px_w, px_h)
    with rasterio.open(raw_path) as src:
        data = src.read(1).astype(np.float32)

    data[data < -100] = np.nan

    with rasterio.open(
        output_path, "w", driver="GTiff",
        height=px_h, width=px_w, count=1,
        dtype="float32", crs="EPSG:28992",
        transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(np.nan_to_num(data, nan=-9999.0), 1)

    raw_path.unlink()
    return px_w, px_h


def download_luchtfoto(bbox: tuple, output_path: Path, px_w: int, px_h: int) -> None:
    """Download luchtfoto (RGB) via ArcGIS MapServer."""
    minx, miny, maxx, maxy = bbox

    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "bboxSR": "28992", "imageSR": "28992",
        "size": f"{px_w},{px_h}",
        "format": "png",
        "f": "image",
    }

    resp = requests.get(LUCHTFOTO_URL, params=params, timeout=120)
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "")
    if "image" not in content_type:
        raise RuntimeError(f"Luchtfoto error: {resp.text[:300]}")

    # Parse image and write as GeoTIFF
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    rgb = np.array(img)  # (H, W, 3)

    # Resize if needed to match DTM dimensions
    if rgb.shape[0] != px_h or rgb.shape[1] != px_w:
        img = img.resize((px_w, px_h), Image.BILINEAR)
        rgb = np.array(img)

    transform = from_bounds(minx, miny, maxx, maxy, px_w, px_h)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        output_path, "w", driver="GTiff",
        height=px_h, width=px_w, count=3,
        dtype="uint8", crs="EPSG:28992",
        transform=transform,
    ) as dst:
        for band in range(3):
            dst.write(rgb[:, :, band], band + 1)


def extract_section(line: LineString, start_m: float, length_m: float) -> LineString:
    """Haal een sectie uit een lijn."""
    end_m = min(start_m + length_m, line.length)
    points = []
    d = start_m
    while d <= end_m:
        pt = line.interpolate(d)
        points.append((pt.x, pt.y))
        d += 2.0
    if len(points) < 2:
        return None
    return LineString(points)


def create_tiles(
    dtm_path: Path, labels_path: Path, rgb_path: Path | None,
    tiles_dir: Path, labels_tiles_dir: Path, rgb_tiles_dir: Path | None,
    tile_size: int = 256, overlap: int = 64,
) -> int:
    """Knip DTM, labels, en optioneel RGB in tiles."""
    tiles_dir.mkdir(parents=True, exist_ok=True)
    labels_tiles_dir.mkdir(parents=True, exist_ok=True)
    if rgb_tiles_dir:
        rgb_tiles_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dtm_path) as src:
        dtm = src.read(1)
        dtm_profile = src.profile.copy()
        transform = src.transform

    with rasterio.open(labels_path) as src:
        labels = src.read(1)

    rgb = None
    if rgb_path and rgb_path.exists():
        with rasterio.open(rgb_path) as src:
            rgb = src.read()  # (3, H, W)

    h, w = dtm.shape
    step = tile_size - overlap
    count = 0

    for y in range(0, h - tile_size + 1, step):
        for x in range(0, w - tile_size + 1, step):
            dtm_tile = dtm[y : y + tile_size, x : x + tile_size]
            lbl_tile = labels[y : y + tile_size, x : x + tile_size]

            valid = (dtm_tile > -100) & (dtm_tile != -9999)
            if valid.sum() < tile_size * tile_size * 0.5:
                continue
            if (lbl_tile > 0).sum() < 50:
                continue

            tile_transform = from_bounds(
                transform.c + x * transform.a,
                transform.f + (y + tile_size) * transform.e,
                transform.c + (x + tile_size) * transform.a,
                transform.f + y * transform.e,
                tile_size, tile_size,
            )

            name = f"tile_{y:04d}_{x:04d}.tif"

            tile_profile = dtm_profile.copy()
            tile_profile.update(height=tile_size, width=tile_size, transform=tile_transform)
            with rasterio.open(tiles_dir / name, "w", **tile_profile) as dst:
                dst.write(dtm_tile, 1)

            lbl_profile = tile_profile.copy()
            lbl_profile.update(dtype="uint8", nodata=0)
            with rasterio.open(labels_tiles_dir / name, "w", **lbl_profile) as dst:
                dst.write(lbl_tile.astype(np.uint8), 1)

            if rgb is not None and rgb_tiles_dir:
                rgb_tile = rgb[:, y : y + tile_size, x : x + tile_size]
                rgb_profile = tile_profile.copy()
                rgb_profile.update(dtype="uint8", count=3, nodata=0)
                with rasterio.open(rgb_tiles_dir / name, "w", **rgb_profile) as dst:
                    dst.write(rgb_tile)

            count += 1

    return count


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Opgeschaalde Training - DTM + Luchtfoto")
    print("=" * 60)

    # Lees trajecten
    with open(GEOJSON_PATH) as f:
        geojson = json.load(f)

    lines = [shape(feat["geometry"]) for feat in geojson["features"]]
    gemeenten = [feat["properties"].get("gemeente", f"traject_{i}")
                 for i, feat in enumerate(geojson["features"])]

    # Directories voor tiles (gecombineerd van alle secties)
    tiles_dir = OUTPUT_DIR / "tiles" / "dtm"
    labels_tiles_dir = OUTPUT_DIR / "tiles" / "labels"
    rgb_tiles_dir = OUTPUT_DIR / "tiles" / "rgb"

    # Wis eventuele oude tiles
    for d in [tiles_dir, labels_tiles_dir, rgb_tiles_dir]:
        if d.exists():
            for f in d.glob("*.tif"):
                f.unlink()

    total_tiles = 0
    section_id = 0

    for traject_idx, (line, gemeente) in enumerate(zip(lines, gemeenten)):
        print(f"\n--- Traject {traject_idx + 1}: {gemeente} ({line.length:.0f}m) ---")

        # Genereer secties langs dit traject
        start = 500  # skip eerste 500m (vaak rommelig bij aansluitingen)
        while start + SECTION_LENGTH_M <= line.length - 500:
            section = extract_section(line, start, SECTION_LENGTH_M)
            if section is None:
                start += SECTION_SPACING_M
                continue

            section_id += 1
            section_dir = OUTPUT_DIR / f"sections" / f"s{section_id:02d}"
            section_dir.mkdir(parents=True, exist_ok=True)

            bbox = section.buffer(BUFFER_M).bounds
            dtm_path = section_dir / "dtm.tif"
            rgb_path = section_dir / "luchtfoto.tif"
            labels_path = section_dir / "labels.tif"

            print(f"\n  Sectie {section_id}: {gemeente} {start:.0f}-{start + SECTION_LENGTH_M:.0f}m")

            # Download DTM
            try:
                print(f"    DTM downloaden...", end=" ", flush=True)
                px_w, px_h = download_ahn4_dtm(bbox, dtm_path)
                print(f"OK ({px_w}x{px_h})")
            except Exception as e:
                print(f"FOUT: {e}")
                start += SECTION_SPACING_M
                continue

            # Download luchtfoto
            try:
                print(f"    Luchtfoto downloaden...", end=" ", flush=True)
                download_luchtfoto(bbox, rgb_path, px_w, px_h)
                print("OK")
            except Exception as e:
                print(f"FOUT: {e}")
                rgb_path = None

            # Genereer labels
            try:
                print(f"    Labels genereren...", end=" ", flush=True)
                generate_training_labels(
                    dtm_path=str(dtm_path),
                    centerline=section,
                    output_labels_path=str(labels_path),
                    crest_buffer=1.5, talud_width=12.0, teen_buffer=2.0,
                )
            except Exception as e:
                print(f"FOUT: {e}")
                start += SECTION_SPACING_M
                continue

            # Knip tiles (met unieke prefix per sectie)
            n = create_tiles(
                dtm_path, labels_path, rgb_path,
                tiles_dir, labels_tiles_dir, rgb_tiles_dir,
                tile_size=TILE_SIZE, overlap=TILE_OVERLAP,
            )

            # Hernoem tiles met sectie-prefix om botsingen te voorkomen
            for d in [tiles_dir, labels_tiles_dir, rgb_tiles_dir]:
                if not d.exists():
                    continue
                for f in sorted(d.glob("tile_*.tif")):
                    new_name = f"s{section_id:02d}_{f.name}"
                    if not (d / new_name).exists():
                        f.rename(d / new_name)

            total_tiles += n
            print(f"    {n} tiles gegenereerd (totaal: {total_tiles})")

            start += SECTION_SPACING_M

    print(f"\n{'=' * 60}")
    print(f"Data-generatie voltooid: {total_tiles} tiles van {section_id} secties")
    print(f"{'=' * 60}")

    if total_tiles < 10:
        print("Te weinig tiles voor training.")
        return

    # Train model
    print(f"\nTrainen U-Net ({EPOCHS} epochs, DTM + slope + RGB)...")
    try:
        import torch
        from kruinlijn.dl.train import train_model

        has_rgb = any(rgb_tiles_dir.glob("*.tif")) if rgb_tiles_dir.exists() else False

        model = train_model(
            tiles_dir=str(tiles_dir),
            labels_dir=str(labels_tiles_dir),
            rgb_dir=str(rgb_tiles_dir) if has_rgb else None,
            output_dir=str(OUTPUT_DIR / "checkpoints"),
            epochs=EPOCHS,
            batch_size=4,
            lr=1e-3,
        )

        # Voorspelling op eerste sectie als test
        print("\nVoorspelling op test-sectie...")
        from kruinlijn.dl.predict import predict_tiles

        test_dtm = OUTPUT_DIR / "sections" / "s01" / "dtm.tif"
        test_rgb = OUTPUT_DIR / "sections" / "s01" / "luchtfoto.tif"
        if test_dtm.exists():
            in_ch = 5 if has_rgb else 2
            predict_tiles(
                dtm_path=str(test_dtm),
                model_path=str(OUTPUT_DIR / "checkpoints" / "best_model.pt"),
                output_path=str(OUTPUT_DIR / "prediction_test.tif"),
                rgb_path=str(test_rgb) if has_rgb and test_rgb.exists() else None,
                in_channels=in_ch,
            )

    except ImportError:
        print("PyTorch niet geinstalleerd. pip install -e '.[dl]'")

    print(f"\nKlaar! Output in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
