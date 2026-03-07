"""End-to-end deep learning pipeline voor dijksegmentatie.

Downloadt AHN4 DTM data, genereert labels via de morfologische methode,
knipt tiles, traint een U-Net, en voorspelt op het volledige DTM.

Gebruik:
    python examples/demo_dl_waal.py
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import LineString, shape
import requests
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
OUTPUT_DIR = Path("output/dl_waal")
DATA_DIR = Path("data/raw")
GEOJSON_PATH = DATA_DIR / "Trajecten ZWO.geojson"

# Sectie: 500m, start op 6km (rustig stuk zonder snelweg/brug)
SECTION_START_M = 6000
SECTION_LENGTH_M = 500
BUFFER_M = 50


def download_ahn4_dtm(bbox: tuple, output_path: Path, size_px: int = 2048) -> None:
    """Download AHN4 DTM via de ArcGIS ImageServer exportImage endpoint.

    Parameters
    ----------
    bbox : tuple
        (minx, miny, maxx, maxy) in EPSG:28992.
    output_path : Path
        Uitvoerpad voor het GeoTIFF.
    size_px : int
        Maximale afmeting in pixels.
    """
    minx, miny, maxx, maxy = bbox
    width = maxx - minx
    height = maxy - miny
    aspect = width / height

    if aspect >= 1:
        px_w = size_px
        px_h = int(size_px / aspect)
    else:
        px_h = size_px
        px_w = int(size_px * aspect)

    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "bboxSR": "28992",
        "imageSR": "28992",
        "size": f"{px_w},{px_h}",
        "format": "tiff",
        "pixelType": "F32",
        "noData": "-9999",
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }

    print(f"  Downloaden AHN4 DTM ({px_w}x{px_h} px)...")
    print(f"  bbox: {minx:.0f},{miny:.0f},{maxx:.0f},{maxy:.0f}")
    resp = requests.get(f"{AHN4_URL}/exportImage", params=params, timeout=120)
    resp.raise_for_status()

    if resp.headers.get("content-type", "").startswith("application/json"):
        error = resp.json()
        raise RuntimeError(f"AHN4 server error: {error}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Schrijf raw TIFF
    raw_path = output_path.with_suffix(".raw.tif")
    raw_path.write_bytes(resp.content)

    # Heropen en schrijf met correcte georeferentie
    transform = from_bounds(minx, miny, maxx, maxy, px_w, px_h)
    with rasterio.open(raw_path) as src:
        data = src.read(1).astype(np.float32)

    # Vervang nodata
    data[data < -100] = np.nan

    with rasterio.open(
        output_path, "w", driver="GTiff",
        height=px_h, width=px_w, count=1,
        dtype="float32", crs="EPSG:28992",
        transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(np.nan_to_num(data, nan=-9999.0), 1)

    raw_path.unlink()
    print(f"  DTM opgeslagen: {output_path} ({px_w}x{px_h}, {width:.0f}x{height:.0f}m)")
    return px_w, px_h


def download_luchtfoto(bbox: tuple, output_path: Path, px_w: int, px_h: int) -> None:
    """Download luchtfoto (RGB) via ArcGIS MapServer."""
    from io import BytesIO
    from PIL import Image

    minx, miny, maxx, maxy = bbox
    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "bboxSR": "28992", "imageSR": "28992",
        "size": f"{px_w},{px_h}",
        "format": "png", "f": "image",
    }
    resp = requests.get(LUCHTFOTO_URL, params=params, timeout=120)
    resp.raise_for_status()

    if "image" not in resp.headers.get("content-type", ""):
        raise RuntimeError(f"Luchtfoto error: {resp.text[:300]}")

    img = Image.open(BytesIO(resp.content)).convert("RGB")
    rgb = np.array(img.resize((px_w, px_h), Image.BILINEAR))

    transform = from_bounds(minx, miny, maxx, maxy, px_w, px_h)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        output_path, "w", driver="GTiff",
        height=px_h, width=px_w, count=3,
        dtype="uint8", crs="EPSG:28992", transform=transform,
    ) as dst:
        for band in range(3):
            dst.write(rgb[:, :, band], band + 1)

    print(f"  Luchtfoto opgeslagen: {output_path}")


def create_tiles(
    dtm_path: Path, labels_path: Path, rgb_path: Path | None,
    tiles_dir: Path, labels_tiles_dir: Path, rgb_tiles_dir: Path | None,
    tile_size: int = 256, overlap: int = 64,
) -> int:
    """Knip DTM, labels, en optioneel RGB in tiles voor training."""
    for d in [tiles_dir, labels_tiles_dir]:
        d.mkdir(parents=True, exist_ok=True)
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
                rgb_prof = tile_profile.copy()
                rgb_prof.update(dtype="uint8", count=3, nodata=0)
                with rasterio.open(rgb_tiles_dir / name, "w", **rgb_prof) as dst:
                    dst.write(rgb_tile)

            count += 1

    return count


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Lees hartlijn uit GeoJSON
    print("=" * 60)
    print("Deep Learning Pipeline - Dijksegmentatie")
    print("=" * 60)

    with open(GEOJSON_PATH) as f:
        geojson = json.load(f)

    full_line = shape(geojson["features"][0]["geometry"])
    print(f"\nTraject: West Maas en Waal, totale lengte: {full_line.length:.0f}m")

    # Neem een sectie
    section_end = min(SECTION_START_M + SECTION_LENGTH_M, full_line.length)
    section_points = []
    d = SECTION_START_M
    while d <= section_end:
        pt = full_line.interpolate(d)
        section_points.append((pt.x, pt.y))
        d += 2.0  # 2m sampling
    centerline = LineString(section_points)
    print(f"Sectie: {SECTION_START_M}m - {section_end:.0f}m ({centerline.length:.0f}m)")

    # 2. Download AHN4 DTM + luchtfoto
    bbox = centerline.buffer(BUFFER_M).bounds
    dtm_path = OUTPUT_DIR / "dtm_waal_section.tif"
    rgb_path = OUTPUT_DIR / "luchtfoto_waal_section.tif"

    print(f"\n[1/6] Downloaden AHN4 DTM...")
    px_w, px_h = download_ahn4_dtm(bbox, dtm_path)

    print(f"\n[2/6] Downloaden luchtfoto...")
    try:
        download_luchtfoto(bbox, rgb_path, px_w, px_h)
        has_rgb = True
    except Exception as e:
        print(f"  Luchtfoto mislukt: {e}")
        has_rgb = False

    # 3. Morfologische kniklijnen
    print(f"\n[3/6] Morfologische kniklijnendetectie...")
    kniklijnen, points_gdf = kniklijnen_pipeline(
        dtm_path=str(dtm_path),
        centerline=centerline,
        output_gpkg=str(OUTPUT_DIR / "kniklijnen_waal.gpkg"),
        spacing=5.0,
        width=60.0,
        smooth_sigma=2.0,
    )
    for naam, lijn in kniklijnen.items():
        status = "Ja" if lijn is not None else "Nee"
        print(f"  {naam:20s}: {status}")

    # 4. Genereer segmentatie-labels
    print(f"\n[4/6] Genereren trainingsdata (labels)...")
    labels_path = OUTPUT_DIR / "labels_waal_section.tif"
    generate_training_labels(
        dtm_path=str(dtm_path),
        centerline=centerline,
        output_labels_path=str(labels_path),
        crest_buffer=1.5,
        talud_width=12.0,
        teen_buffer=2.0,
    )

    # 5. Knip tiles (DTM + labels + RGB)
    print(f"\n[5/6] Knippen van tiles...")
    tiles_dir = OUTPUT_DIR / "tiles" / "dtm"
    labels_tiles_dir = OUTPUT_DIR / "tiles" / "labels"
    rgb_tiles_dir = OUTPUT_DIR / "tiles" / "rgb" if has_rgb else None
    n_tiles = create_tiles(
        dtm_path, labels_path, rgb_path if has_rgb else None,
        tiles_dir, labels_tiles_dir, rgb_tiles_dir,
        tile_size=256, overlap=64,
    )
    print(f"  {n_tiles} tiles gegenereerd" + (" (DTM + RGB)" if has_rgb else " (DTM)"))

    if n_tiles < 4:
        print("  Te weinig tiles voor training. Probeer een grotere sectie.")
        _visualize(dtm_path, kniklijnen, centerline, labels_path, rgb_path=rgb_path if has_rgb else None)
        return

    # 6. Train model
    print(f"\n[6/6] Trainen U-Net model (DTM + slope" + (" + RGB)" if has_rgb else ")") + "...")
    try:
        import torch
        from kruinlijn.dl.train import train_model

        model = train_model(
            tiles_dir=str(tiles_dir),
            labels_dir=str(labels_tiles_dir),
            rgb_dir=str(rgb_tiles_dir) if has_rgb else None,
            output_dir=str(OUTPUT_DIR / "checkpoints"),
            epochs=30,
            batch_size=4,
            lr=1e-3,
        )

        # Voorspelling
        in_ch = 5 if has_rgb else 2
        print(f"\nVoorspelling op volledige DTM ({in_ch} kanalen)...")
        from kruinlijn.dl.predict import predict_tiles

        pred_path = OUTPUT_DIR / "prediction_waal.tif"
        predict_tiles(
            dtm_path=str(dtm_path),
            model_path=str(OUTPUT_DIR / "checkpoints" / "best_model.pt"),
            output_path=str(pred_path),
            rgb_path=str(rgb_path) if has_rgb else None,
            in_channels=in_ch,
        )

        _visualize(dtm_path, kniklijnen, centerline, labels_path, pred_path,
                    rgb_path=rgb_path if has_rgb else None)

    except ImportError:
        print("  PyTorch niet geinstalleerd. Installeer met: pip install -e '.[dl]'")
        _visualize(dtm_path, kniklijnen, centerline, labels_path,
                    rgb_path=rgb_path if has_rgb else None)


def _visualize(
    dtm_path, kniklijnen, centerline, labels_path, pred_path=None, rgb_path=None,
):
    """Maak overzichtsplot."""
    panels = ["dtm", "luchtfoto", "labels"]
    if pred_path:
        panels.append("pred")
    if not rgb_path:
        panels.remove("luchtfoto")

    n_cols = len(panels)
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 7))
    if n_cols == 1:
        axes = [axes]

    kleuren = {
        "kruin": "red", "binnenkruin": "orange", "buitenkruin": "darkorange",
        "binnenberm": "purple", "buitenberm": "mediumpurple",
        "binnenteen": "green", "buitenteen": "blue",
    }

    for i, panel in enumerate(panels):
        ax = axes[i]

        if panel == "dtm":
            with rasterio.open(str(dtm_path)) as src:
                show(src, ax=ax, cmap="terrain", title="DTM + kniklijnen")
            for naam, lijn in kniklijnen.items():
                if lijn is not None and naam in kleuren:
                    x, y = lijn.xy
                    ax.plot(x, y, color=kleuren[naam], linewidth=1.5, label=naam)
            cx, cy = centerline.xy
            ax.plot(cx, cy, "k:", linewidth=0.8, alpha=0.4, label="hartlijn")
            ax.legend(fontsize=6, loc="upper left")

        elif panel == "luchtfoto":
            with rasterio.open(str(rgb_path)) as src:
                show(src, ax=ax, title="Luchtfoto + kniklijnen")
            for naam, lijn in kniklijnen.items():
                if lijn is not None and naam in kleuren:
                    x, y = lijn.xy
                    ax.plot(x, y, color=kleuren[naam], linewidth=1.5)

        elif panel == "labels":
            with rasterio.open(str(labels_path)) as src:
                show(src, ax=ax, cmap="tab10", vmin=0, vmax=6,
                     title="Morfologische labels")

        elif panel == "pred":
            with rasterio.open(str(pred_path)) as src:
                show(src, ax=ax, cmap="tab10", vmin=0, vmax=6,
                     title="DL voorspelling")

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "dl_pipeline_overzicht.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot opgeslagen: {plot_path}")


if __name__ == "__main__":
    main()
