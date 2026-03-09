"""Training met alleen WSRL referentielijnen (binnenkruin, buitenkruin, binnenteen, buitenteen).

Gebruik:
    python examples/train_wsrl.py           # volledige training (50 epochs)
    python examples/train_wsrl.py --quick   # 20 epochs, snelle test
    python examples/train_wsrl.py --train-only  # hergebruik bestaande tiles
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from shapely.geometry import LineString, box
from shapely.ops import linemerge, unary_union


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kruinlijn.pipeline import generate_labels_from_lines
from kruinlijn.data import (
    download_ahn4_dtm,
    download_ahn4_dsm,
    download_luchtfoto,
    download_bgt_waterdeel,
)

# --- Configuratie ---
WSRL_FILES = {
    "binnenkruin": Path("data/raw/Binnenkruinlijn (1).geojson"),
    "buitenkruin": Path("data/raw/Buitenkruinlijn _Referentielijn (WSRL) (1).geojson"),
    "binnenteen":  Path("data/raw/Binnenteen Scope (1).geojson"),
    "buitenteen":  Path("data/raw/Buitenteen Scope (1).geojson"),
}

OUTPUT_DIR = Path("output/dl_wsrl")
TILE_SIZE = 256
TILE_OVERLAP = 64
BUFFER_M = 60
EPOCHS = 70
MAX_DIJKEN = None
SKIP_EXISTING_SECTIONS = True


# --- Helpers (gekopieerd uit train_combined.py) ---

def compute_dijk_sections(dijk_lines: dict, section_width: float = 2000, buffer_m: float = 60):
    """Verdeel een dijk in overlappende secties van ~section_width meter."""
    all_lines = [l for lines in dijk_lines.values() for l in lines]
    if not all_lines:
        return []

    merged = linemerge(unary_union(all_lines))
    if merged.is_empty:
        return []

    main_line = merged if merged.geom_type == "LineString" else max(merged.geoms, key=lambda g: g.length)
    total_length = main_line.length
    step = section_width * 0.75

    sections = []
    pos = 0.0
    while pos < total_length:
        start = max(0.0, pos - buffer_m)
        end = min(total_length, pos + section_width + buffer_m)
        p_start = main_line.interpolate(start)
        p_end = main_line.interpolate(end)

        xs = [p.x for p in [main_line.interpolate(t) for t in np.linspace(start, end, 20)]]
        ys = [p.y for p in [main_line.interpolate(t) for t in np.linspace(start, end, 20)]]
        minx, miny, maxx, maxy = min(xs) - buffer_m, min(ys) - buffer_m, max(xs) + buffer_m, max(ys) + buffer_m

        sections.append((minx, miny, maxx, maxy))
        pos += step
        if pos >= total_length:
            break

    return sections


def clip_lines_to_bbox(lines_dict: dict, bbox: tuple) -> dict:
    """Clip referentielijnen tot bounding box, merge per type tot enkele LineString."""
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


def create_tiles(dtm_path, labels_path, rgb_path, tiles_dir, labels_tiles_dir, rgb_tiles_dir,
                 dsm_path=None, dsm_tiles_dir=None, tile_size=256, overlap=64):
    """Knip DTM, labels, DSM en optioneel RGB in tiles."""
    from rasterio.transform import from_bounds
    for d in [tiles_dir, labels_tiles_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)
    if rgb_tiles_dir:
        Path(rgb_tiles_dir).mkdir(parents=True, exist_ok=True)
    if dsm_tiles_dir:
        Path(dsm_tiles_dir).mkdir(parents=True, exist_ok=True)

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
            dtm_tile = dtm[y:y + tile_size, x:x + tile_size]
            lbl_tile = labels[y:y + tile_size, x:x + tile_size]

            if ((dtm_tile > -100) & (dtm_tile != -9999)).sum() < tile_size * tile_size * 0.5:
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

            with rasterio.open(Path(tiles_dir) / name, "w", **tile_profile) as dst:
                dst.write(dtm_tile, 1)

            lbl_profile = tile_profile.copy()
            lbl_profile.update(dtype="uint8", nodata=0)
            with rasterio.open(Path(labels_tiles_dir) / name, "w", **lbl_profile) as dst:
                dst.write(lbl_tile.astype(np.uint8), 1)

            if rgb is not None and rgb_tiles_dir:
                rgb_tile = rgb[:, y:y + tile_size, x:x + tile_size]
                rgb_prof = tile_profile.copy()
                rgb_prof.update(dtype="uint8", count=3, nodata=0)
                with rasterio.open(Path(rgb_tiles_dir) / name, "w", **rgb_prof) as dst:
                    dst.write(rgb_tile)

            if dsm is not None and dsm_tiles_dir:
                dsm_tile = dsm[y:y + tile_size, x:x + tile_size]
                with rasterio.open(Path(dsm_tiles_dir) / name, "w", **tile_profile) as dst:
                    dst.write(dsm_tile, 1)

            count += 1

    return count


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Training met WSRL referentielijnen")
    print("=" * 60)

    # 1. Laad WSRL lijnen
    import geopandas as gpd
    wsrl_lines: dict[str, list] = {k: [] for k in WSRL_FILES}

    print("\n[1] Laden WSRL kniklijnen...")
    for key, path in WSRL_FILES.items():
        if not path.exists():
            print(f"  WSRL {key}: niet gevonden ({path})")
            continue
        gdf = gpd.read_file(path)
        if gdf.crs and gdf.crs.to_epsg() != 28992:
            gdf = gdf.to_crs("EPSG:28992")
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            parts = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
            for part in parts:
                if part.length > 5:
                    wsrl_lines[key].append(part)
        total_len = sum(l.length for l in wsrl_lines[key])
        print(f"  {key}: {len(wsrl_lines[key])} lijnen, {total_len:.0f}m")

    # WSRL heeft alle lijnen in hetzelfde traject → behandel als 1 dijk
    dijken: dict[str, dict] = {}

    n_types = sum(1 for v in wsrl_lines.values() if v)
    if n_types < 2:
        print("Onvoldoende WSRL lijntypes gevonden (minimaal 2 nodig).")
        return

    dijken["WSRL"] = wsrl_lines
    total_len = sum(l.length for v in wsrl_lines.values() for l in v)
    print(f"\n  WSRL: {n_types} types, {total_len:.0f}m")

    if not dijken:
        print("Geen geldige dijken gevonden.")
        return

    if MAX_DIJKEN:
        dijken = dict(
            sorted(dijken.items(),
                   key=lambda x: sum(l.length for v in x[1].values() for l in v),
                   reverse=True)[:MAX_DIJKEN]
        )
        print(f"  Gelimiteerd tot {MAX_DIJKEN} dijken")

    # 2. Tile directories
    tiles_dir = OUTPUT_DIR / "tiles" / "dtm"
    labels_tiles_dir = OUTPUT_DIR / "tiles" / "labels"
    rgb_tiles_dir = OUTPUT_DIR / "tiles" / "rgb"
    dsm_tiles_dir = OUTPUT_DIR / "tiles" / "dsm"

    total_tiles = 0
    section_id = 0

    # 3. Per dijk: download + labels + tiles
    for dijk_idx, (naam, dijk_lines) in enumerate(dijken.items()):
        total_length = sum(l.length for v in dijk_lines.values() for l in v)
        n_types = sum(1 for v in dijk_lines.values() if v)

        print(f"\n{'=' * 50}")
        print(f"Dijk {dijk_idx+1}/{len(dijken)}: {naam} ({total_length:.0f}m, {n_types} types)")
        print("=" * 50)

        sections = compute_dijk_sections(dijk_lines, section_width=2000, buffer_m=BUFFER_M)
        if not sections:
            print("  Geen secties, skip.")
            continue

        print(f"  {len(sections)} secties")

        for bbox in sections:
            section_id += 1
            section_dir = OUTPUT_DIR / "sections" / f"s{section_id:03d}"
            section_dir.mkdir(parents=True, exist_ok=True)

            dtm_path = section_dir / "dtm.tif"
            dsm_path = section_dir / "dsm.tif"
            rgb_path = section_dir / "luchtfoto.tif"
            labels_path = section_dir / "labels.tif"

            # Caching
            section_tiles = list(tiles_dir.glob(f"s{section_id:03d}_*.tif")) if tiles_dir.exists() else []
            if SKIP_EXISTING_SECTIONS and section_tiles and labels_path.exists():
                total_tiles += len(section_tiles)
                print(f"    Cache: {len(section_tiles)} tiles hergebruikt")
                continue

            # Download DTM
            if dtm_path.exists() and SKIP_EXISTING_SECTIONS:
                with rasterio.open(dtm_path) as src:
                    px_w, px_h = src.width, src.height
                print(f"  DTM: {dtm_path.name} (cached)")
            else:
                try:
                    px_w, px_h = download_ahn4_dtm(bbox, dtm_path)
                except Exception as e:
                    print(f"    DTM FOUT: {e}")
                    continue

            # Download DSM
            if not (dsm_path.exists() and SKIP_EXISTING_SECTIONS):
                try:
                    download_ahn4_dsm(bbox, dsm_path)
                except Exception as e:
                    dsm_path = None
            else:
                print(f"  DSM: {dsm_path.name} (cached)")

            # Download luchtfoto
            if not (rgb_path.exists() and SKIP_EXISTING_SECTIONS):
                try:
                    download_luchtfoto(bbox, rgb_path, px_w, px_h)
                except Exception as e:
                    rgb_path = None
            else:
                print(f"  Luchtfoto: {rgb_path.name} (cached)")

            # BGT sloten
            try:
                from shapely.geometry import shape as shp_shape
                water_features = download_bgt_waterdeel(bbox)
                sloot_polygons = [shp_shape(f["geometry"]) for f in water_features if f.get("geometry")]
            except Exception:
                sloot_polygons = []

            # Clip en check referentielijnen
            clipped = clip_lines_to_bbox(dijk_lines, bbox)
            if sum(1 for v in clipped.values() if v) < 2:
                continue

            # Genereer labels
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
                if not d or not Path(str(d)).exists():
                    continue
                for f in sorted(Path(str(d)).glob("tile_*.tif")):
                    new_name = f"s{section_id:03d}_{f.name}"
                    if not (Path(str(d)) / new_name).exists():
                        f.rename(Path(str(d)) / new_name)

            total_tiles += n

        print(f"  Tiles tot nu toe: {total_tiles}")

    print(f"\n{'=' * 60}")
    print(f"Data-generatie voltooid: {total_tiles} tiles van {section_id} secties")
    print("=" * 60)

    if total_tiles < 10:
        print("Te weinig tiles voor training.")
        return

    # 4. Train model
    print(f"\nTrainen model ({EPOCHS} epochs, WSRL only)...")
    try:
        from kruinlijn.dl.train import train_model

        has_rgb = any(rgb_tiles_dir.glob("*.tif")) if rgb_tiles_dir.exists() else False
        has_dsm = any(dsm_tiles_dir.glob("*.tif")) if dsm_tiles_dir.exists() else False

        train_model(
            tiles_dir=str(tiles_dir),
            labels_dir=str(labels_tiles_dir),
            rgb_dir=str(rgb_tiles_dir) if has_rgb else None,
            dsm_dir=str(dsm_tiles_dir) if has_dsm else None,
            output_dir=str(OUTPUT_DIR / "checkpoints"),
            epochs=EPOCHS,
            batch_size=4,
            lr=1e-3,
        )
    except ImportError:
        print("PyTorch niet geinstalleerd. pip install -e '.[dl]'")

    print(f"\nKlaar! Output in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    if "--quick" in sys.argv:
        MAX_DIJKEN = 5
        EPOCHS = 20
        print(f"Quick mode: {MAX_DIJKEN} dijken, 20 epochs")
    if "--train-only" in sys.argv:
        print("Train-only mode: hergebruik bestaande tiles")
        from kruinlijn.dl.train import train_model
        tiles_dir = OUTPUT_DIR / "tiles" / "dtm"
        labels_dir = OUTPUT_DIR / "tiles" / "labels"
        rgb_dir = OUTPUT_DIR / "tiles" / "rgb"
        dsm_dir = OUTPUT_DIR / "tiles" / "dsm"
        has_rgb = any(rgb_dir.glob("*.tif")) if rgb_dir.exists() else False
        has_dsm = any(dsm_dir.glob("*.tif")) if dsm_dir.exists() else False
        train_model(
            tiles_dir=str(tiles_dir),
            labels_dir=str(labels_dir),
            rgb_dir=str(rgb_dir) if has_rgb else None,
            dsm_dir=str(dsm_dir) if has_dsm else None,
            output_dir=str(OUTPUT_DIR / "checkpoints"),
            epochs=EPOCHS,
            batch_size=4,
            lr=1e-3,
        )
    else:
        main()
