"""Training met DTB referentielijnen + BGT waterdelen.

Gebruikt professioneel ingemeten DTB-lijnen (kruinlijn, talud bovenkant,
talud onderkant) als ground truth, aangevuld met BGT waterdelen voor
sloot/insteek-detectie. Downloadt AHN4 DTM/DSM + luchtfoto + BGT per sectie,
genereert labels, en traint een Attention U-Net.

Gebruik:
    python examples/train_with_dtb.py [dtb_bestand.geojson]

Standaard: data/raw/dtb_kruinlijnen_selectie.geojson
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import LineString, box
from shapely.ops import linemerge, unary_union

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kruinlijn.pipeline import generate_labels_from_lines
from kruinlijn.data import (
    download_ahn4_dtm,
    download_ahn4_dsm,
    download_luchtfoto,
    download_bgt_waterdeel,
    load_dtb_lines,
    classify_dtb_sides,
)

# --- Configuratie ---
DTB_DEFAULT = Path("data/raw/dtb_kruinlijnen_selectie.geojson")
OUTPUT_DIR = Path("output/dl_dtb")

# Training parameters
SECTION_LENGTH_M = 2000
SECTION_SPACING_M = 2500
BUFFER_M = 60
TILE_SIZE = 256
TILE_OVERLAP = 64
EPOCHS = 50


def compute_sections(dtb_lines: dict, section_width: float = 2000, buffer_m: float = 60) -> list[tuple]:
    """Bereken secties (bounding boxes) op basis van de DTB-lijnen.

    Per X-strook wordt de lokale Y-extent van de DTB-lijnen bepaald,
    zodat de bbox strak rond de dijk valt (niet de hele Y-range).

    Returns
    -------
    list[tuple]
        Lijst van (minx, miny, maxx, maxy) bounding boxes.
    """
    all_lines = []
    for lines in dtb_lines.values():
        all_lines.extend(lines)

    if not all_lines:
        return []

    all_x = [c[0] for l in all_lines for c in l.coords]
    min_x, max_x = min(all_x), max(all_x)

    sections = []
    x = min_x
    spacing = section_width * 0.8  # 20% overlap

    while x < max_x:
        x_end = x + section_width
        # Bepaal lokale Y-extent van lijnen in deze X-strook
        local_y = []
        for lines in dtb_lines.values():
            for line in lines:
                for c in line.coords:
                    if x - buffer_m <= c[0] <= x_end + buffer_m:
                        local_y.append(c[1])

        if local_y:
            local_min_y = min(local_y) - buffer_m
            local_max_y = max(local_y) + buffer_m
            bbox = (x - buffer_m, local_min_y, x_end + buffer_m, local_max_y)
            sections.append(bbox)

        x += spacing

    return sections


def clip_lines_to_bbox(
    lines_dict: dict[str, list], bbox: tuple,
) -> dict[str, "LineString | None"]:
    """Clip DTB-lijnen (lijsten) tot bounding box, merge per type tot 1 lijn."""
    clip_box = box(*bbox)
    result = {}

    for key, lines in lines_dict.items():
        clipped_parts = []
        for line in lines:
            clipped = line.intersection(clip_box)
            if clipped.is_empty:
                continue
            if clipped.geom_type == "LineString" and clipped.length > 5:
                clipped_parts.append(clipped)
            elif clipped.geom_type == "MultiLineString":
                for part in clipped.geoms:
                    if part.length > 5:
                        clipped_parts.append(part)

        if not clipped_parts:
            result[key] = None
            continue

        if len(clipped_parts) == 1:
            result[key] = clipped_parts[0]
        else:
            merged = linemerge(clipped_parts)
            if merged.geom_type == "LineString":
                result[key] = merged
            elif merged.geom_type == "MultiLineString":
                result[key] = max(merged.geoms, key=lambda g: g.length)
            else:
                result[key] = max(clipped_parts, key=lambda g: g.length)

    return result



def create_tiles(
    dtm_path: Path, labels_path: Path, rgb_path: Path | None,
    tiles_dir: Path, labels_tiles_dir: Path, rgb_tiles_dir: Path | None,
    dsm_path: Path | None = None, dsm_tiles_dir: Path | None = None,
    tile_size: int = 256, overlap: int = 64,
) -> int:
    """Knip DTM, labels, DSM en optioneel RGB in tiles."""
    tiles_dir.mkdir(parents=True, exist_ok=True)
    labels_tiles_dir.mkdir(parents=True, exist_ok=True)
    if rgb_tiles_dir:
        rgb_tiles_dir.mkdir(parents=True, exist_ok=True)
    if dsm_tiles_dir:
        dsm_tiles_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dtm_path) as src:
        dtm = src.read(1)
        dtm_profile = src.profile.copy()
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
                rgb_prof = tile_profile.copy()
                rgb_prof.update(dtype="uint8", count=3, nodata=0)
                with rasterio.open(rgb_tiles_dir / name, "w", **rgb_prof) as dst:
                    dst.write(rgb_tile)

            if dsm is not None and dsm_tiles_dir:
                dsm_tile = dsm[y : y + tile_size, x : x + tile_size]
                with rasterio.open(dsm_tiles_dir / name, "w", **tile_profile) as dst:
                    dst.write(dsm_tile, 1)

            count += 1

    return count


def main():
    # DTB bestand
    dtb_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DTB_DEFAULT
    if not dtb_path.exists():
        print(f"DTB bestand niet gevonden: {dtb_path}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Training met DTB referentielijnen + BGT waterdelen")
    print("=" * 60)

    # 1. Laad DTB lijnen
    print("\n[1] Laden DTB lijnen...")
    dtb_lines = load_dtb_lines(dtb_path)

    # 2. Classificeer binnen/buiten-zijde
    print("\n[2] Classificeren binnen/buiten...")
    ref_lines = classify_dtb_sides(dtb_lines)
    for key, lines in ref_lines.items():
        if lines:
            total = sum(l.length for l in lines)
            print(f"  {key}: {len(lines)} lijnen, {total:.0f}m")

    # 3. Bereken secties
    print("\n[3] Secties berekenen...")
    sections = compute_sections(ref_lines, section_width=SECTION_LENGTH_M, buffer_m=BUFFER_M)
    print(f"  {len(sections)} secties over het DTB-gebied")

    # 4. Directories voor tiles
    tiles_dir = OUTPUT_DIR / "tiles" / "dtm"
    labels_tiles_dir = OUTPUT_DIR / "tiles" / "labels"
    rgb_tiles_dir = OUTPUT_DIR / "tiles" / "rgb"
    dsm_tiles_dir = OUTPUT_DIR / "tiles" / "dsm"

    for d in [tiles_dir, labels_tiles_dir, rgb_tiles_dir, dsm_tiles_dir]:
        if d.exists():
            for f in d.glob("*.tif"):
                f.unlink()

    total_tiles = 0
    section_id = 0

    print(f"\n--- DTB-gebied ({len(sections)} secties) ---")

    for bbox in sections:
        section_id += 1
        section_dir = OUTPUT_DIR / "sections" / f"s{section_id:02d}"
        section_dir.mkdir(parents=True, exist_ok=True)

        dtm_path = section_dir / "dtm.tif"
        dsm_path = section_dir / "dsm.tif"
        rgb_path = section_dir / "luchtfoto.tif"
        labels_path = section_dir / "labels.tif"

        print(f"\n  Sectie {section_id}: X {bbox[0]:.0f}-{bbox[2]:.0f}")

        # Download DTM
        try:
            print(f"    DTM downloaden...", end=" ", flush=True)
            px_w, px_h = download_ahn4_dtm(bbox, dtm_path)
            print(f"OK ({px_w}x{px_h})")
        except Exception as e:
            print(f"FOUT: {e}")
            continue

        # Download DSM
        try:
            print(f"    DSM downloaden...", end=" ", flush=True)
            download_ahn4_dsm(bbox, dsm_path)
            print("OK")
        except Exception as e:
            print(f"FOUT: {e}")
            dsm_path = None

        # Download luchtfoto
        try:
            print(f"    Luchtfoto downloaden...", end=" ", flush=True)
            download_luchtfoto(bbox, rgb_path, px_w, px_h)
            print("OK")
        except Exception as e:
            print(f"FOUT: {e}")
            rgb_path = None

        # Download BGT waterdelen (sloten)
        print(f"    BGT waterdelen ophalen...", end=" ", flush=True)
        try:
            water_features = download_bgt_waterdeel(bbox)
            sloot_polygons = [f["geometry"] for f in water_features]
            print(f"{len(sloot_polygons)} polygonen")
        except Exception as e:
            print(f"FOUT: {e}")
            sloot_polygons = []

        # Clip referentielijnen tot deze sectie
        print(f"    Labels genereren...")
        clipped = clip_lines_to_bbox(ref_lines, bbox)
        for key, line in clipped.items():
            if line is not None:
                print(f"      {key}: {line.length:.0f}m")

        n_available = sum(1 for v in clipped.values() if v is not None)
        if n_available < 2:
            print(f"    Te weinig referentielijnen ({n_available}), skip sectie.")
            continue

        # Genereer labels (met BGT sloten)
        try:
            generate_labels_from_lines(
                dtm_path=str(dtm_path),
                reference_lines=clipped,
                output_labels_path=str(labels_path),
                kruin_buffer=2.0,
                teen_buffer=2.5,
                sloot_polygons=sloot_polygons,
            )
        except Exception as e:
            print(f"    Labels FOUT: {e}")
            continue

        # Knip tiles
        n = create_tiles(
            dtm_path, labels_path, rgb_path,
            tiles_dir, labels_tiles_dir, rgb_tiles_dir,
            dsm_path=dsm_path, dsm_tiles_dir=dsm_tiles_dir,
            tile_size=TILE_SIZE, overlap=TILE_OVERLAP,
        )

        # Hernoem met sectie-prefix
        for d in [tiles_dir, labels_tiles_dir, rgb_tiles_dir, dsm_tiles_dir]:
            if not d.exists():
                continue
            for f in sorted(d.glob("tile_*.tif")):
                new_name = f"s{section_id:02d}_{f.name}"
                if not (d / new_name).exists():
                    f.rename(d / new_name)

        total_tiles += n
        print(f"    {n} tiles gegenereerd (totaal: {total_tiles})")

    print(f"\n{'=' * 60}")
    print(f"Data-generatie voltooid: {total_tiles} tiles van {section_id} secties")
    print(f"{'=' * 60}")

    if total_tiles < 10:
        print("Te weinig tiles voor training.")
        return

    # Train model
    print(f"\nTrainen Attention U-Net ({EPOCHS} epochs, DTB labels + BGT)...")
    try:
        from kruinlijn.dl.train import train_model

        has_rgb = any(rgb_tiles_dir.glob("*.tif")) if rgb_tiles_dir.exists() else False
        has_dsm = any(dsm_tiles_dir.glob("*.tif")) if dsm_tiles_dir.exists() else False

        model = train_model(
            tiles_dir=str(tiles_dir),
            labels_dir=str(labels_tiles_dir),
            rgb_dir=str(rgb_tiles_dir) if has_rgb else None,
            dsm_dir=str(dsm_tiles_dir) if has_dsm else None,
            output_dir=str(OUTPUT_DIR / "checkpoints"),
            epochs=EPOCHS,
            batch_size=4,
            lr=1e-3,
        )

        # Test voorspelling
        print("\nVoorspelling op test-sectie...")
        from kruinlijn.dl.predict import predict_tiles

        test_dtm = OUTPUT_DIR / "sections" / "s01" / "dtm.tif"
        test_dsm = OUTPUT_DIR / "sections" / "s01" / "dsm.tif"
        test_rgb = OUTPUT_DIR / "sections" / "s01" / "luchtfoto.tif"
        if test_dtm.exists():
            in_ch = 5
            if has_dsm:
                in_ch += 1
            if has_rgb:
                in_ch += 3
            predict_tiles(
                dtm_path=str(test_dtm),
                model_path=str(OUTPUT_DIR / "checkpoints" / "best_model.pt"),
                output_path=str(OUTPUT_DIR / "prediction_test.tif"),
                rgb_path=str(test_rgb) if has_rgb and test_rgb.exists() else None,
                dsm_path=str(test_dsm) if has_dsm and test_dsm.exists() else None,
                in_channels=in_ch,
            )

    except ImportError:
        print("PyTorch niet geinstalleerd. pip install -e '.[dl]'")

    print(f"\nKlaar! Output in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
