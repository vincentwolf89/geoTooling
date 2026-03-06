"""Demo: kruinlijndetectie via morfologische analyse.

Gebruik:
    python examples/demo_morpho.py data/raw/dtm.tif data/raw/hartlijn.gpkg output/

Vereist:
    - Een DTM GeoTIFF van het dijkgebied
    - Een GeoPackage met een hartlijn (LineString) van de dijk
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.plot import show

# Voeg src toe aan het pad
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kruinlijn.pipeline import morpho_pipeline


def main():
    if len(sys.argv) < 4:
        print("Gebruik: python demo_morpho.py <dtm.tif> <hartlijn.gpkg> <output_dir/>")
        sys.exit(1)

    dtm_path = sys.argv[1]
    hartlijn_path = sys.argv[2]
    output_dir = Path(sys.argv[3])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lees hartlijn
    gdf = gpd.read_file(hartlijn_path)
    centerline = gdf.geometry.iloc[0]

    # Detecteer kruinlijn
    crest_line, points_gdf = morpho_pipeline(
        dtm_path=dtm_path,
        centerline=centerline,
        output_gpkg=output_dir / "kruinlijn_resultaat.gpkg",
        spacing=5.0,
        width=40.0,
        smooth_sigma=2.0,
        method="curvature",
    )

    # Visualisatie
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Links: DTM met kruinlijn
    with rasterio.open(dtm_path) as src:
        show(src, ax=axes[0], cmap="terrain", title="DTM met kruinlijn")
    if crest_line is not None:
        x, y = crest_line.xy
        axes[0].plot(x, y, "r-", linewidth=2, label="Kruinlijn")
    cx, cy = centerline.xy
    axes[0].plot(cx, cy, "b--", linewidth=1, alpha=0.5, label="Hartlijn")
    axes[0].legend()

    # Rechts: een voorbeeld-dwarsprofiel
    from kruinlijn.morpho import generate_cross_profiles, extract_profile_elevations, detect_crest_points

    profiles = generate_cross_profiles(centerline, spacing=5.0, width=40.0)
    profiles = extract_profile_elevations(profiles, dtm_path)
    profiles = detect_crest_points(profiles, method="curvature")

    # Kies een profiel uit het midden
    mid = len(profiles) // 2
    p = profiles[mid]
    axes[1].plot(p["offsets"], p["elevations"], "k-", label="Hoogteprofiel")
    if p["crest_idx"] is not None:
        axes[1].axvline(p["offsets"][p["crest_idx"]], color="r", linestyle="--", label="Kruin")
        axes[1].plot(p["offsets"][p["crest_idx"]], p["crest_z"], "ro", markersize=10)
    axes[1].set_xlabel("Afstand op dwarsprofiel (m)")
    axes[1].set_ylabel("Hoogte (m NAP)")
    axes[1].set_title(f"Dwarsprofiel op {p['distance']:.0f}m")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "kruinlijn_analyse.png", dpi=150)
    plt.show()
    print(f"Plot opgeslagen: {output_dir / 'kruinlijn_analyse.png'}")


if __name__ == "__main__":
    main()
