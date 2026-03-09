"""Plot voorspellingsresultaten met AHN hillshade + luchtfoto achtergrond.

Gebruik:
    python examples/plot_prediction.py <predict_dir> <output_png>

Voorbeeld:
    python examples/plot_prediction.py output/predict_zwo_10km output/predict_zwo_10km/result.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import rasterio
from rasterio.merge import merge
from rasterio.transform import array_bounds
import geopandas as gpd
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Klasse-namen en kleuren (10 klassen)
CLASS_NAMES = [
    "achtergrond", "kruin", "talud_binnen", "binnenberm",
    "teen_binnen", "talud_buiten", "buitenberm", "teen_buiten",
    "insteek", "sloot",
]
CLASS_COLORS = [
    "#00000000",  # 0 achtergrond — transparant
    "#e74c3c",    # 1 kruin — rood
    "#3498db",    # 2 talud_binnen — blauw
    "#2ecc71",    # 3 binnenberm — groen
    "#1abc9c",    # 4 teen_binnen — teal
    "#f39c12",    # 5 talud_buiten — oranje
    "#f1c40f",    # 6 buitenberm — geel
    "#e67e22",    # 7 teen_buiten — donkeroranje
    "#9b59b6",    # 8 insteek — paars
    "#34495e",    # 9 sloot — donkerblauw/grijs
]

# Lijnkleuren voor vectorlijnen — key moet voorkomen in de lijnnaam
LINE_COLORS = {
    "binnenkruin": "#e74c3c",    # rood
    "buitenkruin": "#c0392b",    # donkerrood
    "binnenteen":  "#1abc9c",    # teal
    "buitenteen":  "#e67e22",    # oranje
    "kruinlijn":   "#e74c3c",    # rood (fallback voor losse kruinlijn)
    "insteek":     "#9b59b6",    # paars
    "sloot":       "#34495e",    # donkergrijs
    "binnenberm":  "#2ecc71",    # groen
    "buitenberm":  "#f1c40f",    # geel
}


def hillshade(dtm: np.ndarray, azimuth: float = 315, altitude: float = 45) -> np.ndarray:
    """Bereken hillshade van DTM array."""
    azimuth_rad = np.radians(360 - azimuth)
    altitude_rad = np.radians(altitude)

    # Gradienten
    dy, dx = np.gradient(dtm)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)

    hs = (np.sin(altitude_rad) * np.cos(slope) +
          np.cos(altitude_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect))
    hs = np.clip(hs, 0, 1)
    return hs


def load_mosaic(section_dirs: list[Path], filename: str):
    """Merge meerdere GeoTIFFs tot één mozaïek."""
    paths = [d / filename for d in section_dirs if (d / filename).exists()]
    if not paths:
        return None, None

    datasets = [rasterio.open(p) for p in paths]
    mosaic, transform = merge(datasets)
    crs = datasets[0].crs
    nodata = datasets[0].nodata
    for ds in datasets:
        ds.close()
    return mosaic, transform, crs, nodata


def main():
    if len(sys.argv) < 2:
        print("Gebruik: python plot_prediction.py <predict_dir> [output_png]")
        sys.exit(1)

    predict_dir = Path(sys.argv[1])
    output_png = Path(sys.argv[2]) if len(sys.argv) > 2 else predict_dir / "result.png"

    section_dirs = sorted(predict_dir.glob("section_*"))
    if not section_dirs:
        # Misschien is predict_dir zelf de sectie
        section_dirs = [predict_dir]

    print(f"  {len(section_dirs)} secties gevonden")

    # -- DTM mozaïek --
    print("  DTM laden...")
    dtm_result = load_mosaic(section_dirs, "dtm.tif")
    dtm_arr = dtm_result[0][0].astype(np.float32)
    dtm_transform = dtm_result[1]
    dtm_valid = dtm_arr.copy()
    dtm_valid[(dtm_valid < -10) | (dtm_valid > 100)] = np.nan

    # -- Luchtfoto mozaïek --
    print("  Luchtfoto laden...")
    rgb_result = load_mosaic(section_dirs, "luchtfoto.tif")
    if rgb_result is not None:
        rgb_arr = rgb_result[0][:3].astype(np.float32)
        # Normaliseer naar 0-1
        for c in range(3):
            p2, p98 = np.nanpercentile(rgb_arr[c], [2, 98])
            if p98 > p2:
                rgb_arr[c] = np.clip((rgb_arr[c] - p2) / (p98 - p2), 0, 1)
        rgb_img = np.moveaxis(rgb_arr, 0, -1)
    else:
        rgb_img = None

    # -- Voorspelling mozaïek --
    print("  Voorspelling laden...")
    pred_result = load_mosaic(section_dirs, "prediction.tif")
    pred_arr = pred_result[0][0].astype(np.int32)

    # -- Vectorlijnen --
    print("  Vectorlijnen laden...")
    gdfs = []
    for sd in section_dirs:
        gpkg = sd / "kniklijnen.gpkg"
        if gpkg.exists():
            try:
                gdf = gpd.read_file(gpkg)
                if len(gdf) > 0:
                    gdfs.append(gdf)
            except Exception:
                pass
    lines_gdf = gpd.pd.concat(gdfs, ignore_index=True) if gdfs else None

    # -- Hillshade --
    print("  Hillshade berekenen...")
    hs = hillshade(np.where(np.isnan(dtm_valid), 0, dtm_valid))

    # -- Plotting --
    print("  Plot genereren...")
    h, w = pred_arr.shape

    fig, axes = plt.subplots(1, 2, figsize=(24, 10), dpi=150)

    # Extent voor imshow (in RD meter)
    bounds = array_bounds(h, w, dtm_transform)
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

    # ---- Links: AHN hillshade + voorspelling overlay ----
    ax = axes[0]
    ax.set_title("Dijkclassificatie — AHN hillshade", fontsize=13, fontweight="bold")

    # Hillshade achtergrond
    ax.imshow(hs, extent=extent, cmap="gray", vmin=0.2, vmax=1.0,
              aspect="equal", origin="upper", interpolation="bilinear")

    # DTM kleur overlay (licht)
    dtm_norm = (dtm_valid - np.nanpercentile(dtm_valid, 5)) / (np.nanpercentile(dtm_valid, 95) - np.nanpercentile(dtm_valid, 5) + 1e-6)
    dtm_norm = np.clip(dtm_norm, 0, 1)
    ax.imshow(dtm_norm, extent=extent, cmap="terrain", alpha=0.35,
              aspect="equal", origin="upper", interpolation="bilinear")

    # Segmentatie overlay
    cmap = ListedColormap(CLASS_COLORS)
    seg_rgba = np.zeros((*pred_arr.shape, 4), dtype=np.float32)
    for cls_idx, color in enumerate(CLASS_COLORS):
        if color == "#00000000":
            continue
        mask = pred_arr == cls_idx
        if not mask.any():
            continue
        r = int(color[1:3], 16) / 255
        g = int(color[3:5], 16) / 255
        b = int(color[5:7], 16) / 255
        seg_rgba[mask] = [r, g, b, 0.55]

    ax.imshow(seg_rgba, extent=extent, aspect="equal", origin="upper", interpolation="nearest")

    # Vectorlijnen over hillshade
    if lines_gdf is not None and len(lines_gdf) > 0:
        for _, row in lines_gdf.iterrows():
            naam = row.get("naam", "")
            color = "#ff0000"
            for key, col in LINE_COLORS.items():
                if key.lower() in naam.lower():
                    color = col
                    break
            if row.geometry and not row.geometry.is_empty:
                xs, ys = row.geometry.xy
                ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.9)

    ax.set_xlabel("RD X (m)", fontsize=9)
    ax.set_ylabel("RD Y (m)", fontsize=9)
    ax.tick_params(labelsize=7)

    # ---- Rechts: Luchtfoto + vectorlijnen ----
    ax2 = axes[1]
    ax2.set_title("Dijkkniklijnen — Luchtfoto", fontsize=13, fontweight="bold")

    if rgb_img is not None:
        ax2.imshow(rgb_img, extent=extent, aspect="equal", origin="upper", interpolation="bilinear")
    else:
        ax2.imshow(hs, extent=extent, cmap="gray", aspect="equal", origin="upper")

    if lines_gdf is not None and len(lines_gdf) > 0:
        for _, row in lines_gdf.iterrows():
            naam = row.get("naam", "")
            color = "#ff0000"
            lw = 1.5
            for key, col in LINE_COLORS.items():
                if key.lower() in naam.lower():
                    color = col
                    break
            if row.geometry and not row.geometry.is_empty:
                xs, ys = row.geometry.xy
                ax2.plot(xs, ys, color=color, linewidth=lw, alpha=0.95)

    ax2.set_xlabel("RD X (m)", fontsize=9)
    ax2.tick_params(labelsize=7)

    # Legenda
    legend_patches = []
    for cls_idx, (name, color) in enumerate(zip(CLASS_NAMES[1:], CLASS_COLORS[1:]), 1):
        patch = mpatches.Patch(color=color, label=name, alpha=0.8)
        legend_patches.append(patch)

    # Lijnlegenda
    line_handles = []
    for naam, color in LINE_COLORS.items():
        line_handles.append(plt.Line2D([0], [0], color=color, lw=2, label=naam))

    axes[0].legend(handles=legend_patches, loc="upper right", fontsize=7,
                   framealpha=0.85, ncol=2, title="Klassen", title_fontsize=8)
    axes[1].legend(handles=line_handles, loc="upper right", fontsize=7,
                   framealpha=0.85, title="Vectorlijnen", title_fontsize=8)

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, bbox_inches="tight", dpi=150)
    print(f"\n  Plot opgeslagen: {output_png}")
    plt.close()


if __name__ == "__main__":
    main()
