"""Quick test: DL training met puntenwolk-DTM ipv server-DTM.

Gebruikt een stuk ZWO traject met WSRL referentielijnen.
Vergelijkt tiles/features uit puntenwolk vs server.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import LineString, box

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kruinlijn.data import (
    download_ahn4_dtm,
    download_ahn4_dsm,
    download_ahn4_pointcloud_dtm,
    download_ahn4_pointcloud_dsm,
    download_luchtfoto,
    download_bgt_waterdeel,
)
from kruinlijn.pipeline import generate_labels_from_lines

# --- Config ---
HDSR_FILE = Path("data/raw/Kniklijnen HDSR.geojson")
OUTPUT_DIR = Path("output/dl_pointcloud_test")
TILE_SIZE = 256
TILE_OVERLAP = 64

# Test dijk: Voorboezemkade Haanwijk West (compact, 4 types, ~945m)
TEST_DIJK = "Voorboezemkade Haanwijk West"
# Bbox met 80m buffer rondom de dijk
TEST_BBOX = (124310, 456360, 124650, 456710)


def load_hdsr_lines_for_dijk(dijk_naam, bbox):
    """Laad HDSR kniklijnen voor een specifieke dijk, clip naar bbox."""
    import json
    from pyproj import Transformer
    from shapely.geometry import shape as shp_shape
    from shapely.ops import linemerge

    clip_box = box(*bbox)
    to_rd = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
    type_map = {15: "binnenteen", 16: "buitenteen", 17: "binnenkruin", 18: "buitenkruin"}

    with open(HDSR_FILE) as f:
        data = json.load(f)

    result = {}
    for feat in data["features"]:
        props = feat["properties"]
        if props.get("naam") != dijk_naam:
            continue
        target = type_map.get(props.get("type"))
        if target is None:
            continue

        geom = shp_shape(feat["geometry"])
        if geom.is_empty:
            continue

        parts = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        for part in parts:
            coords = [to_rd.transform(x, y) for x, y in part.coords]
            line = LineString(coords)
            clipped = line.intersection(clip_box)
            if clipped.is_empty or clipped.length < 5:
                continue
            if clipped.geom_type == "MultiLineString":
                clipped = max(clipped.geoms, key=lambda g: g.length)
            if target not in result:
                result[target] = clipped
            else:
                merged = linemerge([result[target], clipped])
                if merged.geom_type == "MultiLineString":
                    merged = max(merged.geoms, key=lambda g: g.length)
                result[target] = merged

    for key, line in result.items():
        print(f"  HDSR {key}: {line.length:.0f}m")
    return result


def create_tiles(dtm_path, labels_path, rgb_path, dsm_path, prefix, tiles_dir, labels_dir, rgb_dir, dsm_dir):
    """Knip rasters in tiles, return aantal."""
    for d in [tiles_dir, labels_dir, rgb_dir, dsm_dir]:
        d.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dtm_path) as src:
        dtm = src.read(1)
        profile = src.profile.copy()
        transform = src.transform

    with rasterio.open(labels_path) as src:
        labels = src.read(1)

    rgb = None
    if rgb_path and Path(rgb_path).exists():
        with rasterio.open(rgb_path) as src:
            rgb = src.read()

    dsm = None
    if dsm_path and Path(dsm_path).exists():
        with rasterio.open(dsm_path) as src:
            dsm = src.read(1)

    h, w = dtm.shape
    step = TILE_SIZE - TILE_OVERLAP
    count = 0

    for y in range(0, h - TILE_SIZE + 1, step):
        for x in range(0, w - TILE_SIZE + 1, step):
            dtm_tile = dtm[y:y + TILE_SIZE, x:x + TILE_SIZE]
            lbl_tile = labels[y:y + TILE_SIZE, x:x + TILE_SIZE]

            valid = (dtm_tile > -100) & (dtm_tile != -9999)
            if valid.sum() < TILE_SIZE * TILE_SIZE * 0.5:
                continue
            if (lbl_tile > 0).sum() < 50:
                continue

            tile_transform = from_bounds(
                transform.c + x * transform.a,
                transform.f + (y + TILE_SIZE) * transform.e,
                transform.c + (x + TILE_SIZE) * transform.a,
                transform.f + y * transform.e,
                TILE_SIZE, TILE_SIZE,
            )

            name = f"{prefix}_tile_{y:04d}_{x:04d}.tif"
            tp = profile.copy()
            tp.update(height=TILE_SIZE, width=TILE_SIZE, transform=tile_transform)

            with rasterio.open(tiles_dir / name, "w", **tp) as dst:
                dst.write(dtm_tile, 1)

            lp = tp.copy()
            lp.update(dtype="uint8", nodata=0)
            with rasterio.open(labels_dir / name, "w", **lp) as dst:
                dst.write(lbl_tile.astype(np.uint8), 1)

            if rgb is not None:
                rgb_tile = rgb[:, y:y + TILE_SIZE, x:x + TILE_SIZE]
                rp = tp.copy()
                rp.update(dtype="uint8", count=3, nodata=0)
                with rasterio.open(rgb_dir / name, "w", **rp) as dst:
                    dst.write(rgb_tile)

            if dsm is not None:
                dsm_tile = dsm[y:y + TILE_SIZE, x:x + TILE_SIZE]
                with rasterio.open(dsm_dir / name, "w", **tp) as dst:
                    dst.write(dsm_tile, 1)

            count += 1

    return count


def compare_dtm_features(server_dtm_path, pc_dtm_path):
    """Vergelijk terreinfeatures (slope, curvature) tussen twee DTMs."""
    for label, path in [("Server", server_dtm_path), ("Puntenwolk", pc_dtm_path)]:
        with rasterio.open(path) as src:
            dtm = src.read(1).astype(float)
            res = abs(src.transform.a)

        valid = (dtm > -100) & (dtm != -9999)
        dtm[~valid] = np.nan

        dy, dx = np.gradient(dtm, res)
        slope = np.sqrt(dx**2 + dy**2)
        slope_deg = np.degrees(np.arctan(slope))

        dyy, _ = np.gradient(dy, res)
        _, dxx = np.gradient(dx, res)
        curv = -(dxx + dyy)

        print(f"\n  {label} DTM ({res:.2f}m):")
        print(f"    Hoogte: {np.nanmin(dtm):.1f} - {np.nanmax(dtm):.1f}m")
        print(f"    Slope:  gem={np.nanmean(slope_deg):.1f}°, max={np.nanmax(slope_deg):.1f}°")
        print(f"    Curv:   std={np.nanstd(curv):.4f} (hoger = scherpere knikken)")
        print(f"    NaN:    {np.isnan(dtm).sum()} pixels ({np.isnan(dtm).mean():.1%})")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bbox = TEST_BBOX

    print("=" * 60)
    print("DL Training test: Puntenwolk vs Server DTM")
    print(f"Test area: {bbox}")
    print("=" * 60)

    # 1. Laad HDSR labels
    print(f"\n[1] HDSR referentielijnen laden ({TEST_DIJK})...")
    ref_lines = load_hdsr_lines_for_dijk(TEST_DIJK, bbox)
    if len(ref_lines) < 2:
        print("Te weinig referentielijnen in test-area!")
        return

    # 2. Download BGT
    print("\n[2] BGT waterdelen...")
    try:
        water = download_bgt_waterdeel(bbox)
        sloot_polys = [f["geometry"] for f in water]
    except Exception:
        sloot_polys = []

    # 3. Download luchtfoto (gedeeld)
    rgb_path = OUTPUT_DIR / "luchtfoto.tif"

    # 4a. Server DTM/DSM
    print("\n[3a] Server DTM/DSM downloaden...")
    server_dir = OUTPUT_DIR / "server"
    server_dir.mkdir(parents=True, exist_ok=True)
    server_dtm = server_dir / "dtm.tif"
    server_dsm = server_dir / "dsm.tif"

    if not server_dtm.exists():
        px_w, px_h = download_ahn4_dtm(bbox, server_dtm)
    else:
        with rasterio.open(server_dtm) as src:
            px_w, px_h = src.width, src.height
        print(f"  DTM: cached ({px_w}x{px_h})")

    if not server_dsm.exists():
        download_ahn4_dsm(bbox, server_dsm)
    else:
        print("  DSM: cached")

    if not rgb_path.exists():
        download_luchtfoto(bbox, rgb_path, px_w, px_h)
    else:
        print("  Luchtfoto: cached")

    # 4b. Puntenwolk DTM/DSM
    print("\n[3b] Puntenwolk DTM/DSM...")
    pc_dir = OUTPUT_DIR / "pointcloud"
    pc_dir.mkdir(parents=True, exist_ok=True)
    pc_dtm = pc_dir / "dtm.tif"
    pc_dsm = pc_dir / "dsm.tif"

    if not pc_dtm.exists():
        pc_w, pc_h = download_ahn4_pointcloud_dtm(bbox, pc_dtm, resolution=0.5)
    else:
        with rasterio.open(pc_dtm) as src:
            pc_w, pc_h = src.width, src.height
        print(f"  PC-DTM: cached ({pc_w}x{pc_h})")

    if not pc_dsm.exists():
        pc_dw, pc_dh = download_ahn4_pointcloud_dsm(bbox, pc_dsm, resolution=0.5)
    else:
        print("  PC-DSM: cached")

    # 5. Vergelijk DTM features
    print("\n[4] Feature vergelijking...")
    compare_dtm_features(server_dtm, pc_dtm)

    # 6. Genereer labels + tiles voor beide
    for label, dtm_path, dsm_path in [
        ("server", server_dtm, server_dsm),
        ("pc", pc_dtm, pc_dsm),
    ]:
        print(f"\n[5-{label}] Labels + tiles genereren ({label})...")
        labels_path = OUTPUT_DIR / label / "labels.tif"
        labels_path.parent.mkdir(parents=True, exist_ok=True)

        generate_labels_from_lines(
            dtm_path=str(dtm_path),
            reference_lines=ref_lines,
            output_labels_path=str(labels_path),
            kruin_buffer=2.0,
            teen_buffer=2.5,
            sloot_polygons=sloot_polys,
        )

        # Maak luchtfoto met zelfde dimensies als DTM
        with rasterio.open(dtm_path) as src:
            dtm_w, dtm_h = src.width, src.height
        local_rgb = OUTPUT_DIR / label / "luchtfoto.tif"
        if not local_rgb.exists():
            download_luchtfoto(bbox, local_rgb, dtm_w, dtm_h)

        n_tiles = create_tiles(
            dtm_path, labels_path, local_rgb, dsm_path,
            prefix=label,
            tiles_dir=OUTPUT_DIR / "tiles" / label / "dtm",
            labels_dir=OUTPUT_DIR / "tiles" / label / "labels",
            rgb_dir=OUTPUT_DIR / "tiles" / label / "rgb",
            dsm_dir=OUTPUT_DIR / "tiles" / label / "dsm",
        )
        print(f"  {n_tiles} tiles gegenereerd ({label})")

    # 7. Quick train op beide
    print("\n[6] Training (10 epochs per variant)...")
    try:
        from kruinlijn.dl.train import train_model

        for label in ["server", "pc"]:
            tiles_dir = OUTPUT_DIR / "tiles" / label / "dtm"
            labels_dir = OUTPUT_DIR / "tiles" / label / "labels"
            rgb_dir = OUTPUT_DIR / "tiles" / label / "rgb"
            dsm_dir = OUTPUT_DIR / "tiles" / label / "dsm"

            n = len(list(tiles_dir.glob("*.tif"))) if tiles_dir.exists() else 0
            if n < 2:
                print(f"  {label}: te weinig tiles ({n}), skip training")
                continue

            print(f"\n  Training {label} ({n} tiles, 10 epochs)...")
            model = train_model(
                tiles_dir=str(tiles_dir),
                labels_dir=str(labels_dir),
                rgb_dir=str(rgb_dir) if any(rgb_dir.glob("*.tif")) else None,
                dsm_dir=str(dsm_dir) if any(dsm_dir.glob("*.tif")) else None,
                output_dir=str(OUTPUT_DIR / "checkpoints" / label),
                epochs=10,
                batch_size=4,
                lr=1e-3,
            )

    except ImportError:
        print("PyTorch niet beschikbaar. pip install -e '.[dl]'")

    print(f"\nKlaar! Output in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
