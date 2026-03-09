"""Training met gecombineerde datasets: HDSR kniklijnen + DTB + WSRL + BGT.

Combineert professioneel ingemeten referentielijnen uit meerdere bronnen:
- HDSR kniklijnen: kruin, teen, berm, insteek (8 types, 98 dijken)
- DTB (RWS): kruinlijn, talud bovenkant/onderkant
- WSRL: binnenkruin, buitenkruin, binnenteen, buitenteen
- BGT waterdelen: sloten en waterlopen

Downloadt AHN4 DTM/DSM + luchtfoto + BGT per sectie, genereert labels,
en traint een Attention U-Net.

Gebruik:
    python examples/train_combined.py
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
    load_hdsr_kniklijnen,
    load_dtb_lines,
    classify_dtb_sides,
)

# --- Configuratie ---
HDSR_FILE = Path("data/raw/Kniklijnen HDSR.geojson")
DTB_FILE = Path("data/raw/dtb_kruinlijnen_selectie.geojson")
WSRL_FILES = {
    "binnenkruin": Path("data/raw/Binnenkruinlijn (1).geojson"),
    "buitenkruin": Path("data/raw/Buitenkruinlijn _Referentielijn (WSRL) (1).geojson"),
    "binnenteen": Path("data/raw/Binnenteen Scope (1).geojson"),
    "buitenteen": Path("data/raw/Buitenteen Scope (1).geojson"),
}
OUTPUT_DIR = Path("output/dl_combined")

# Training parameters
TILE_SIZE = 256
TILE_OVERLAP = 64
BUFFER_M = 60
EPOCHS = 50
MAX_DIJKEN = None  # None = alle dijken, of een getal om te limiten
SKIP_EXISTING_SECTIONS = True  # Hergebruik al gedownloade secties


def compute_dijk_sections(
    ref_lines: dict[str, list],
    section_width: float = 2000,
    buffer_m: float = 60,
) -> list[tuple]:
    """Bereken secties op basis van referentielijnen extent."""
    all_lines = []
    for lines in ref_lines.values():
        all_lines.extend(lines)
    if not all_lines:
        return []

    all_x = [c[0] for l in all_lines for c in l.coords]
    min_x, max_x = min(all_x), max(all_x)

    # Als het gebied kleiner is dan een sectie, gebruik 1 sectie
    if max_x - min_x < section_width:
        all_y = [c[1] for l in all_lines for c in l.coords]
        return [(min_x - buffer_m, min(all_y) - buffer_m,
                 max_x + buffer_m, max(all_y) + buffer_m)]

    sections = []
    x = min_x
    spacing = section_width * 0.8

    while x < max_x:
        x_end = x + section_width
        local_y = []
        for lines in ref_lines.values():
            for line in lines:
                for c in line.coords:
                    if x - buffer_m <= c[0] <= x_end + buffer_m:
                        local_y.append(c[1])

        if local_y:
            bbox = (x - buffer_m, min(local_y) - buffer_m,
                    x_end + buffer_m, max(local_y) + buffer_m)
            sections.append(bbox)
        x += spacing

    return sections


def clip_lines_to_bbox(
    lines_dict: dict[str, list], bbox: tuple,
) -> dict[str, "LineString | None"]:
    """Clip referentielijnen (lijsten) tot bounding box, merge per type."""
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


def group_by_dijk(ref_lines: dict[str, list], naam_per_line: dict) -> dict[str, dict[str, list]]:
    """Groepeer referentielijnen per dijk-naam voor sectie-berekening."""
    dijken = {}
    for key, lines in ref_lines.items():
        for line in lines:
            line_id = id(line)
            naam = naam_per_line.get(line_id, "onbekend")
            if naam not in dijken:
                dijken[naam] = {k: [] for k in ref_lines.keys()}
            dijken[naam][key].append(line)
    return dijken


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Training met gecombineerde datasets (HDSR + WSRL + DTB + BGT)")
    print("=" * 60)

    # 1. Laad HDSR kniklijnen
    if not HDSR_FILE.exists():
        print(f"HDSR bestand niet gevonden: {HDSR_FILE}")
        sys.exit(1)

    print("\n[1] Laden HDSR kniklijnen...")
    ref_lines = load_hdsr_kniklijnen(HDSR_FILE)

    # Groepeer per dijk-naam voor betere sectie-indeling
    import json
    with open(HDSR_FILE) as f:
        hdsr_data = json.load(f)

    from pyproj import Transformer
    to_rd = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
    from shapely.geometry import shape as shp_shape

    type_map = {
        15: "binnenteen", 16: "buitenteen",
        17: "binnenkruin", 18: "buitenkruin",
        60: "binnenberm", 61: "buitenberm",
        62: "insteek", 63: "insteek",
    }

    # Bouw per-dijk data
    dijken = {}
    for feat in hdsr_data['features']:
        props = feat['properties']
        naam = props.get('naam') or 'onbekend'
        line_type = props.get('type')
        target_key = type_map.get(line_type)
        if target_key is None:
            continue

        geom = shp_shape(feat['geometry'])
        if geom.is_empty:
            continue

        parts = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        for part in parts:
            coords_rd = [to_rd.transform(x, y) for x, y in part.coords]
            line = LineString(coords_rd)
            if line.length < 5:
                continue

            if naam not in dijken:
                dijken[naam] = {k: [] for k in ref_lines.keys()}
            dijken[naam][target_key].append(line)

    # 1b. Laad WSRL referentielijnen
    wsrl_keys = ["binnenkruin", "buitenkruin", "binnenteen", "buitenteen"]
    wsrl_loaded = False
    for wsrl_key, wsrl_path in WSRL_FILES.items():
        if not wsrl_path.exists():
            print(f"  WSRL {wsrl_key}: bestand niet gevonden ({wsrl_path})")
            continue

        import geopandas as _gpd
        wsrl_gdf = _gpd.read_file(wsrl_path)
        if wsrl_gdf.crs and wsrl_gdf.crs.to_epsg() != 28992:
            wsrl_gdf = wsrl_gdf.to_crs("EPSG:28992")

        target_key = {
            "binnenkruin": "binnenkruin",
            "buitenkruin": "buitenkruin",
            "binnenteen": "binnenteen",
            "buitenteen": "buitenteen",
        }[wsrl_key]

        wsrl_lines = []
        for _, row in wsrl_gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            parts = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
            for part in parts:
                if part.length > 5:
                    wsrl_lines.append(part)

        if wsrl_lines:
            total_len = sum(l.length for l in wsrl_lines)
            print(f"  WSRL {target_key}: {len(wsrl_lines)} lijnen, {total_len:.0f}m")

            # Voeg toe aan dijken dict onder naam "WSRL"
            if "WSRL" not in dijken:
                dijken["WSRL"] = {k: [] for k in ref_lines.keys()}
            dijken["WSRL"][target_key].extend(wsrl_lines)
            wsrl_loaded = True

    if wsrl_loaded:
        print("  WSRL data toegevoegd aan trainingsset")

    # 1c. Laad DTB referentielijnen
    if DTB_FILE.exists():
        print("\n[1c] Laden DTB lijnen...")
        dtb_raw = load_dtb_lines(DTB_FILE)
        dtb_classified = classify_dtb_sides(dtb_raw)

        # Groepeer DTB lijnen per lokale cluster (gebruik x-range als proxy)
        dtb_all_lines = []
        for key, lines in dtb_classified.items():
            dtb_all_lines.extend(lines)

        if dtb_all_lines:
            # Verdeel DTB in "dijken" op basis van ruimtelijke nabijheid
            from shapely.ops import unary_union
            from shapely.geometry import MultiLineString

            # Simpele groepering: alles als 1 dijk (DTB data is al per traject)
            dijken["DTB"] = {k: [] for k in ref_lines.keys()}
            for key in ["binnenkruin", "buitenkruin", "binnenteen", "buitenteen"]:
                if key in dtb_classified and dtb_classified[key]:
                    dijken["DTB"][key] = dtb_classified[key]
            print(f"  DTB data toegevoegd aan trainingsset")
    else:
        print(f"  DTB bestand niet gevonden: {DTB_FILE}")

    # Filter dijken met minimaal 2 types
    valid_dijken = {
        naam: lines for naam, lines in dijken.items()
        if sum(1 for v in lines.values() if v) >= 2
    }
    print(f"\n  {len(valid_dijken)} dijken met >= 2 lijn-types (HDSR + WSRL + DTB)")

    if MAX_DIJKEN:
        # Sorteer op totale lengte, pak de langste
        valid_dijken = dict(
            sorted(valid_dijken.items(),
                   key=lambda x: sum(l.length for v in x[1].values() for l in v),
                   reverse=True)[:MAX_DIJKEN]
        )
        print(f"  Gelimiteerd tot {MAX_DIJKEN} dijken")

    # 2. Directories voor tiles
    tiles_dir = OUTPUT_DIR / "tiles" / "dtm"
    labels_tiles_dir = OUTPUT_DIR / "tiles" / "labels"
    rgb_tiles_dir = OUTPUT_DIR / "tiles" / "rgb"
    dsm_tiles_dir = OUTPUT_DIR / "tiles" / "dsm"

    if not SKIP_EXISTING_SECTIONS:
        for d in [tiles_dir, labels_tiles_dir, rgb_tiles_dir, dsm_tiles_dir]:
            if d.exists():
                for f in d.glob("*.tif"):
                    f.unlink()

    total_tiles = 0
    section_id = 0

    # 3. Per dijk: secties berekenen en data genereren
    for dijk_idx, (naam, dijk_lines) in enumerate(valid_dijken.items()):
        total_length = sum(l.length for v in dijk_lines.values() for l in v)
        n_types = sum(1 for v in dijk_lines.values() if v)

        print(f"\n{'=' * 50}")
        print(f"Dijk {dijk_idx+1}/{len(valid_dijken)}: {naam} ({total_length:.0f}m, {n_types} types)")
        print("=" * 50)

        # Bereken secties voor deze dijk
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

            # Check of sectie al verwerkt is (caching)
            section_tiles = list(tiles_dir.glob(f"s{section_id:03d}_*.tif"))
            if SKIP_EXISTING_SECTIONS and section_tiles and labels_path.exists():
                n = len(section_tiles)
                total_tiles += n
                print(f"    Cache: {n} tiles hergebruikt")
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

            # Download BGT waterdelen
            try:
                water_features = download_bgt_waterdeel(bbox)
                sloot_polygons = [f["geometry"] for f in water_features]
            except Exception:
                sloot_polygons = []

            # Clip referentielijnen
            clipped = clip_lines_to_bbox(dijk_lines, bbox)
            n_available = sum(1 for v in clipped.values() if v is not None)
            if n_available < 2:
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
                if not d.exists():
                    continue
                for f in sorted(d.glob("tile_*.tif")):
                    new_name = f"s{section_id:03d}_{f.name}"
                    if not (d / new_name).exists():
                        f.rename(d / new_name)

            total_tiles += n

        print(f"  Tiles tot nu toe: {total_tiles}")

    print(f"\n{'=' * 60}")
    print(f"Data-generatie voltooid: {total_tiles} tiles van {section_id} secties")
    print(f"({'=' * 60})")

    if total_tiles < 10:
        print("Te weinig tiles voor training.")
        return

    # Train model
    print(f"\nTrainen Attention U-Net ({EPOCHS} epochs, HDSR + WSRL + DTB + BGT)...")
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

    except ImportError:
        print("PyTorch niet geinstalleerd. pip install -e '.[dl]'")

    print(f"\nKlaar! Output in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    if "--quick" in sys.argv:
        MAX_DIJKEN = 5
        EPOCHS = 20
        print(f"Quick mode: {MAX_DIJKEN} dijken, {EPOCHS} epochs")
        main()
    elif "--train-only" in sys.argv:
        # Alleen training, skip data-generatie (gebruik bestaande tiles)
        print("Training-only mode: hergebruik bestaande tiles")
        tiles_dir = OUTPUT_DIR / "tiles" / "dtm"
        labels_tiles_dir = OUTPUT_DIR / "tiles" / "labels"
        rgb_tiles_dir = OUTPUT_DIR / "tiles" / "rgb"
        dsm_tiles_dir = OUTPUT_DIR / "tiles" / "dsm"

        n_tiles = len(list(tiles_dir.glob("*.tif"))) if tiles_dir.exists() else 0
        print(f"  {n_tiles} tiles gevonden")

        if n_tiles < 10:
            print("Te weinig tiles. Draai eerst zonder --train-only.")
            sys.exit(1)

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
        print(f"\nKlaar! Output in: {OUTPUT_DIR.resolve()}")
    else:
        main()
