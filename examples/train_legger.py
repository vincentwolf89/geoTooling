"""Training / fine-tuning met WSRL Legger profielkniklijnen.

Laadt binnenkruinlijn, buitenkruinlijn, binnenteenlijn, buitenteenlijn
uit de gedownloade legger GeoJSONs, clustert segmenten tot dijken,
genereert tiles en traint (of fine-tunet) het model.

Gebruik:
    python examples/train_legger.py               # volle run, 50 epochs
    python examples/train_legger.py --quick        # 5 dijken, 20 epochs
    python examples/train_legger.py --train-only   # skip data, hergebruik tiles
    python examples/train_legger.py --finetune     # start vanaf huidig best_model.pt
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, box, shape as shp_shape
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
LEGGER_DIR = Path("data/raw/wsrl_legger")
WDODELTA_DIR = Path("data/raw/wdodelta")
WSRL_MODEL = Path("output/dl_wsrl/checkpoints/best_model.pt")
WSRL_TILES_DIR = Path("output/dl_wsrl/tiles")  # bestaande WSRL tiles
OUTPUT_DIR = Path("output/dl_combined_v2")
TILE_SIZE = 256
TILE_OVERLAP = 64
BUFFER_M = 60
EPOCHS = 100
MAX_DIJKEN = None       # None = alle, of getal voor subset
CLUSTER_BUFFER = 80     # meter — segmenten binnen deze afstand = zelfde dijk
SKIP_EXISTING_SECTIONS = True


# --- Legger laden ---

def load_legger_lines(source: str = "wdodelta") -> dict[str, list[LineString]]:
    """Laad kniklijnen per type als Shapely LineStrings (EPSG:28992).

    source: 'wdodelta' of 'legger' (wsrl legger)
    """
    if source == "wdodelta":
        mapping = {
            "binnenkruin": WDODELTA_DIR / "wdodelta_binnenkruinlijn.geojson",
            "buitenkruin": WDODELTA_DIR / "wdodelta_buitenkruinlijn.geojson",
            "binnenteen":  WDODELTA_DIR / "wdodelta_binnenteenlijn.geojson",
            "buitenteen":  WDODELTA_DIR / "wdodelta_buitenteenlijn.geojson",
        }
    else:
        mapping = {
            "binnenkruin": LEGGER_DIR / "wsrl_legger_binnenkruinlijn.geojson",
            "buitenkruin": LEGGER_DIR / "wsrl_legger_buitenkruinlijn.geojson",
            "binnenteen":  LEGGER_DIR / "wsrl_legger_binnenteenlijn.geojson",
            "buitenteen":  LEGGER_DIR / "wsrl_legger_buitenteenlijn.geojson",
        }
    result = {}
    for key, path in mapping.items():
        with open(path) as f:
            data = json.load(f)
        lines = []
        for feat in data["features"]:
            geom = shp_shape(feat["geometry"])
            if geom.geom_type == "LineString":
                lines.append(geom)
            elif geom.geom_type == "MultiLineString":
                lines.extend(geom.geoms)
        result[key] = lines
        print(f"  {key}: {len(lines)} segmenten geladen")
    return result


def cluster_dijken(lines: dict[str, list[LineString]], cluster_buffer: float = 80.0) -> list[dict]:
    """Groepeer legger-segmenten tot individuele dijken via ruimtelijke clustering.

    Gebruikt binnenkruinlijn als basis: segmenten die overlappen na buffering
    worden samengevoegd tot één dijk. Vervolgens worden alle andere lijntypes
    per dijk verzameld.

    Returns list van dijk-dicts met keys: bbox, binnenkruin, buitenkruin, binnenteen, buitenteen
    """
    from scipy.spatial import cKDTree

    basis = lines["binnenkruin"]
    if not basis:
        return []

    # Maak buffer-polygonen voor binnenkruin segmenten
    buffers = [seg.buffer(cluster_buffer) for seg in basis]

    # Connected-components via union van overlappende buffers
    print(f"  Clusteren {len(buffers)} binnenkruin-segmenten...")
    merged = list(unary_union(buffers).geoms) if unary_union(buffers).geom_type == "MultiPolygon" else [unary_union(buffers)]

    dijken = []
    for cluster_poly in merged:
        bbox = cluster_poly.bounds  # (minx, miny, maxx, maxy)
        bbox_box = box(*bbox)

        dijk = {"bbox": bbox}
        for key, segs in lines.items():
            matching = [s for s in segs if s.intersects(bbox_box)]
            if matching:
                merged_line = linemerge(matching)
                if merged_line.geom_type == "MultiLineString":
                    # neem de totale geometrie als lijst
                    dijk[key] = list(merged_line.geoms)
                else:
                    dijk[key] = [merged_line]
            else:
                dijk[key] = []

        # Alleen dijken met minstens kruin + één teen
        if dijk["binnenkruin"] and (dijk["binnenteen"] or dijk["buitenteen"]):
            dijken.append(dijk)

    print(f"  {len(dijken)} dijken gevonden na clustering")
    return dijken


def clip_lines_to_bbox(dijk: dict, bbox: tuple) -> dict[str, LineString | None]:
    """Clip lijnen van één dijk tot een bbox, merge tot enkele LineString."""
    clip_box = box(*bbox)
    result = {}
    for key in ["binnenkruin", "buitenkruin", "binnenteen", "buitenteen"]:
        segs = dijk.get(key, [])
        parts = []
        for seg in segs:
            clipped = seg.intersection(clip_box)
            if clipped.is_empty:
                continue
            if clipped.geom_type == "LineString" and clipped.length > 5:
                parts.append(clipped)
            elif clipped.geom_type == "MultiLineString":
                parts.extend(g for g in clipped.geoms if g.length > 5)
        if parts:
            merged = linemerge(parts)
            result[key] = merged if merged.geom_type == "LineString" else max(merged.geoms, key=lambda g: g.length)
        else:
            result[key] = None
    return result



def process_dijk_section(
    dijk: dict,
    section_idx: int,
    bbox: tuple,
    sections_dir: Path,
    tiles_dir: Path,
    labels_dir: Path,
    rgb_dir: Path,
    dsm_dir: Path,
):
    """Download data + genereer tiles voor één dijksectie (sla op als .tif)."""
    import rasterio
    from rasterio.transform import from_bounds

    section_dir = sections_dir / f"section_{section_idx:04d}"
    section_dir.mkdir(parents=True, exist_ok=True)

    dtm_path = section_dir / "dtm.tif"
    dsm_path = section_dir / "dsm.tif"
    rgb_path = section_dir / "luchtfoto.tif"
    label_path = section_dir / "labels.tif"

    # Check al verwerkt
    existing = list(tiles_dir.glob(f"section_{section_idx:04d}_*.tif"))
    if SKIP_EXISTING_SECTIONS and existing:
        return 0

    # Download DTM
    try:
        px_w, px_h = download_ahn4_dtm(bbox, dtm_path)
    except Exception as e:
        print(f"    DTM fout: {e}")
        return 0

    has_dsm = False
    try:
        download_ahn4_dsm(bbox, dsm_path)
        has_dsm = True
    except Exception:
        pass

    has_rgb = False
    try:
        download_luchtfoto(bbox, rgb_path, px_w, px_h)
        has_rgb = True
    except Exception:
        pass

    # BGT sloten
    sloot_polygons = []
    try:
        water_features = download_bgt_waterdeel(bbox)
        sloot_polygons = [shp_shape(f["geometry"]) for f in water_features if f.get("geometry")]
    except Exception:
        pass

    # Labels genereren
    ref_lines = clip_lines_to_bbox(dijk, bbox)
    if not any(v is not None for v in ref_lines.values()):
        return 0

    try:
        generate_labels_from_lines(
            dtm_path, ref_lines, label_path, sloot_polygons=sloot_polygons
        )
    except Exception as e:
        print(f"    Label fout: {e}")
        return 0

    # Tiles knippen als .tif (zelfde structuur als train_wsrl.py)
    with rasterio.open(dtm_path) as src:
        dtm_arr = src.read(1)
        dtm_profile = src.profile.copy()
        transform = src.transform

    with rasterio.open(label_path) as src:
        labels_arr = src.read(1)

    rgb_arr = None
    if has_rgb:
        with rasterio.open(rgb_path) as src:
            rgb_arr = src.read()

    dsm_arr = None
    if has_dsm:
        with rasterio.open(dsm_path) as src:
            dsm_arr = src.read(1)

    h, w = dtm_arr.shape
    step = TILE_SIZE - TILE_OVERLAP
    count = 0

    for y in range(0, h - TILE_SIZE + 1, step):
        for x in range(0, w - TILE_SIZE + 1, step):
            dtm_tile = dtm_arr[y:y + TILE_SIZE, x:x + TILE_SIZE]
            lbl_tile = labels_arr[y:y + TILE_SIZE, x:x + TILE_SIZE]

            if ((dtm_tile > -100) & (dtm_tile != -9999)).sum() < TILE_SIZE * TILE_SIZE * 0.5:
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
            name = f"section_{section_idx:04d}_{y:04d}_{x:04d}.tif"
            tile_profile = dtm_profile.copy()
            tile_profile.update(height=TILE_SIZE, width=TILE_SIZE, transform=tile_transform)

            with rasterio.open(tiles_dir / name, "w", **tile_profile) as dst:
                dst.write(dtm_tile, 1)

            lbl_profile = tile_profile.copy()
            lbl_profile.update(dtype="uint8", nodata=0)
            with rasterio.open(labels_dir / name, "w", **lbl_profile) as dst:
                dst.write(lbl_tile.astype(np.uint8), 1)

            if rgb_arr is not None:
                rgb_tile = rgb_arr[:, y:y + TILE_SIZE, x:x + TILE_SIZE]
                rgb_prof = tile_profile.copy()
                rgb_prof.update(dtype="uint8", count=3, nodata=0)
                with rasterio.open(rgb_dir / name, "w", **rgb_prof) as dst:
                    dst.write(rgb_tile)

            if dsm_arr is not None:
                dsm_tile = dsm_arr[y:y + TILE_SIZE, x:x + TILE_SIZE]
                with rasterio.open(dsm_dir / name, "w", **tile_profile) as dst:
                    dst.write(dsm_tile, 1)

            count += 1

    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="5 dijken, 20 epochs")
    parser.add_argument("--train-only", action="store_true", help="skip data-generatie")
    parser.add_argument("--finetune", action="store_true", help="start vanaf bestaand WSRL model")
    args = parser.parse_args()

    epochs = EPOCHS
    max_dijken = MAX_DIJKEN
    finetune_path = None

    if args.quick:
        epochs = 20
        max_dijken = 5

    if args.finetune and WSRL_MODEL.exists():
        finetune_path = WSRL_MODEL
        print(f"Fine-tunen vanaf: {finetune_path}")

    # Dirs aanmaken
    sections_dir = OUTPUT_DIR / "sections"
    tiles_dir = OUTPUT_DIR / "tiles"
    labels_dir = OUTPUT_DIR / "labels"
    rgb_dir = OUTPUT_DIR / "rgb"
    dsm_dir = OUTPUT_DIR / "dsm"
    for d in [sections_dir, tiles_dir, labels_dir, rgb_dir, dsm_dir]:
        d.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from kruinlijn.dl.train import train_model

    if not args.train_only:
        print("\n[1] WDODelta kniklijnen laden...")
        all_lines = load_legger_lines(source="wdodelta")

        print("\n[2] Dijken clusteren...")
        dijken = cluster_dijken(all_lines, CLUSTER_BUFFER)

        if max_dijken:
            dijken = dijken[:max_dijken]
            print(f"  Beperkt tot {max_dijken} dijken")

        print(f"\n[3] Data downloaden + tiles genereren ({len(dijken)} dijken)...")
        total_tiles = 0
        section_idx = 0

        for dijk_idx, dijk in enumerate(dijken):
            bbox = dijk["bbox"]
            # Verdeel in secties van 2km
            minx, miny, maxx, maxy = bbox
            x = minx
            while x < maxx:
                x_end = min(x + 2000, maxx)
                sub_bbox = (x - BUFFER_M, miny - BUFFER_M, x_end + BUFFER_M, maxy + BUFFER_M)
                n = process_dijk_section(
                    dijk, section_idx, sub_bbox,
                    sections_dir, tiles_dir, labels_dir, rgb_dir, dsm_dir
                )
                if n > 0:
                    total_tiles += n
                    if section_idx % 10 == 0:
                        print(f"  [{dijk_idx+1}/{len(dijken)}] sectie {section_idx}: +{n} tiles (totaal {total_tiles})")
                section_idx += 1
                x += 1800  # 200m overlap

        print(f"\n  Totaal: {total_tiles} nieuwe tiles")
    else:
        print("Train-only mode: hergebruik bestaande tiles")

    # Kopieer WSRL tiles erbij (als ze bestaan en nog niet gekopieerd)
    wsrl_dtm = WSRL_TILES_DIR / "dtm"
    if wsrl_dtm.exists():
        import shutil
        wsrl_labels = WSRL_TILES_DIR / "labels"
        wsrl_rgb = WSRL_TILES_DIR / "rgb"
        wsrl_dsm = WSRL_TILES_DIR / "dsm"
        copied = 0
        for src_file in wsrl_dtm.glob("*.tif"):
            dst = tiles_dir / f"wsrl_{src_file.name}"
            if not dst.exists():
                shutil.copy2(src_file, dst)
                # Labels
                lbl_src = wsrl_labels / src_file.name
                if lbl_src.exists():
                    shutil.copy2(lbl_src, labels_dir / f"wsrl_{src_file.name}")
                # RGB
                rgb_src = wsrl_rgb / src_file.name
                if rgb_src.exists():
                    shutil.copy2(rgb_src, rgb_dir / f"wsrl_{src_file.name}")
                # DSM
                dsm_src = wsrl_dsm / src_file.name
                if dsm_src.exists():
                    shutil.copy2(dsm_src, dsm_dir / f"wsrl_{src_file.name}")
                copied += 1
        if copied > 0:
            print(f"  WSRL tiles gekopieerd: {copied}")

    # Tiles tellen
    n_tiles = len(list(tiles_dir.glob("*.tif")))
    has_rgb = any(rgb_dir.glob("*.tif"))
    has_dsm = any(dsm_dir.glob("*.tif"))
    print(f"\n[4] Trainen model ({epochs} epochs, {n_tiles} tiles — WDODelta + WSRL)...")

    train_model(
        tiles_dir=tiles_dir,
        labels_dir=labels_dir,
        rgb_dir=rgb_dir if has_rgb else None,
        dsm_dir=dsm_dir if has_dsm else None,
        output_dir=OUTPUT_DIR / "checkpoints",
        epochs=epochs,
        batch_size=4,
        lr=2e-4 if finetune_path else 1e-3,  # lagere LR voor fine-tuning
        pretrained_path=finetune_path,
    )

    print(f"\nKlaar! Model in: {OUTPUT_DIR / 'checkpoints'}")


if __name__ == "__main__":
    main()
