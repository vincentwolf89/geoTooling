"""Voorspelling op een nieuw dijkgebied.

Downloadt AHN4 DTM + luchtfoto voor een opgegeven traject,
draait het getrainde model, en exporteert vectorlijnen als GeoPackage.

Gebruik:
    python examples/predict_area.py <traject.geojson> [model.pt] [output_dir]

Voorbeeld:
    python examples/predict_area.py data/raw/mijn_traject.geojson output/dl_real_labels/checkpoints/best_model.pt output/predict/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, shape

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kruinlijn.data import download_ahn4_dtm, download_ahn4_dsm, download_luchtfoto
from kruinlijn.dl.predict import predict_tiles
from kruinlijn.dl.vectorize import extract_lines

# Defaults
DEFAULT_MODEL = Path("output/dl_real_labels/checkpoints/best_model.pt")
DEFAULT_OUTPUT = Path("output/predict")
BUFFER_M = 80
IN_CHANNELS = 9  # DTM + slope + aspect + curvature + TPI + nDSM + R + G + B


def load_traject(geojson_path: Path) -> list[LineString]:
    """Laad traject(en) uit GeoJSON. Ondersteunt EPSG:28992 en EPSG:4326."""
    with open(geojson_path) as f:
        data = json.load(f)

    lines = []
    needs_transform = False

    for feat in data["features"]:
        geom = shape(feat["geometry"])
        if geom.geom_type == "LineString":
            lines.append(geom)
        elif geom.geom_type == "MultiLineString":
            lines.extend(geom.geoms)

    if not lines:
        raise ValueError(f"Geen lijnen gevonden in {geojson_path}")

    # Detecteer CRS: als coördinaten < 1000 dan is het WGS84
    sample_x = lines[0].coords[0][0]
    if abs(sample_x) < 1000:
        print("  CRS gedetecteerd: EPSG:4326, transformeren naar 28992...")
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
        transformed = []
        for line in lines:
            coords = [t.transform(x, y) for x, y in line.coords]
            transformed.append(LineString(coords))
        lines = transformed

    total_length = sum(l.length for l in lines)
    print(f"  {len(lines)} traject(en) geladen, totaal {total_length:.0f}m")
    return lines


def predict_traject(
    traject: LineString,
    model_path: Path,
    output_dir: Path,
    section_length: float = 1000,
    section_overlap: float = 200,
) -> list[Path]:
    """Voorspel een heel traject in secties.

    Returns lijst met voorspellings-GeoTIFFs.
    """
    prediction_paths = []
    start = 0.0

    while start < traject.length:
        end = min(start + section_length, traject.length)

        # Maak sectie-lijn
        points = []
        d = start
        while d <= end:
            pt = traject.interpolate(d)
            points.append((pt.x, pt.y))
            d += 2.0
        if len(points) < 2:
            break
        section = LineString(points)

        section_id = len(prediction_paths) + 1
        section_dir = output_dir / f"section_{section_id:03d}"
        section_dir.mkdir(parents=True, exist_ok=True)

        bbox = section.buffer(BUFFER_M).bounds
        dtm_path = section_dir / "dtm.tif"
        dsm_path = section_dir / "dsm.tif"
        rgb_path = section_dir / "luchtfoto.tif"
        pred_path = section_dir / "prediction.tif"

        print(f"\n  Sectie {section_id}: {start:.0f}-{end:.0f}m")

        # Download DTM
        try:
            px_w, px_h = download_ahn4_dtm(bbox, dtm_path)
        except Exception as e:
            print(f"    DTM FOUT: {e}")
            start += section_length - section_overlap
            continue

        # Download DSM
        has_dsm = False
        try:
            download_ahn4_dsm(bbox, dsm_path)
            has_dsm = True
        except Exception as e:
            print(f"    DSM FOUT: {e}")

        # Download luchtfoto
        has_rgb = False
        try:
            download_luchtfoto(bbox, rgb_path, px_w, px_h)
            has_rgb = True
        except Exception as e:
            print(f"    Luchtfoto FOUT: {e}")

        # Voorspelling — kanalen: DTM(1) + slope(1) + aspect(1) + curvature(1) + TPI(1) + nDSM(1) + RGB(3)
        in_ch = 5  # basis terrein kanalen
        if has_dsm:
            in_ch += 1
        if has_rgb:
            in_ch += 3
        predict_tiles(
            dtm_path=str(dtm_path),
            model_path=str(model_path),
            output_path=str(pred_path),
            rgb_path=str(rgb_path) if has_rgb else None,
            dsm_path=str(dsm_path) if has_dsm else None,
            in_channels=in_ch,
        )
        prediction_paths.append(pred_path)

        start += section_length - section_overlap

    return prediction_paths


def main():
    if len(sys.argv) < 2:
        print("Gebruik: python predict_area.py <traject.geojson> [model.pt] [output_dir]")
        sys.exit(1)

    geojson_path = Path(sys.argv[1])
    model_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_MODEL
    output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_OUTPUT

    if not model_path.exists():
        print(f"Model niet gevonden: {model_path}")
        print("Train eerst met: python examples/train_with_real_labels.py")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Dijksegmentatie — Voorspelling op nieuw gebied")
    print("=" * 60)

    # 1. Laad traject
    print("\n[1] Laden traject...")
    trajecten = load_traject(geojson_path)

    all_predictions = []

    for i, traject in enumerate(trajecten):
        print(f"\n{'=' * 40}")
        print(f"Traject {i + 1}/{len(trajecten)} ({traject.length:.0f}m)")
        print("=" * 40)

        traject_dir = output_dir / f"traject_{i + 1:02d}" if len(trajecten) > 1 else output_dir
        predictions = predict_traject(traject, model_path, traject_dir)
        all_predictions.extend(predictions)

    # 2. Vectorlijnen extraheren per sectie
    print(f"\n{'=' * 60}")
    print(f"Vectorlijnen extraheren uit {len(all_predictions)} secties...")

    for pred_path in all_predictions:
        gpkg_path = pred_path.with_name("kniklijnen.gpkg")
        try:
            extract_lines(str(pred_path), str(gpkg_path))
        except Exception as e:
            print(f"  Vectorisatie fout voor {pred_path.name}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Klaar! Output in: {output_dir.resolve()}")
    print(f"GeoPackages met vectorlijnen in elke sectie-map.")
    print("=" * 60)


if __name__ == "__main__":
    main()
