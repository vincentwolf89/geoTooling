"""Test: AHN4 puntenwolk -> 25cm DTM -> knikpunt detectie.

Vergelijkt resultaten van puntenwolk-DTM (25cm) met standaard AHN4 DTM (50cm).
Gebruikt een klein stukje van het ZWO traject (West Maas en Waal).
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path

import geopandas as gpd
import laspy
import rasterio
from rasterio.transform import from_bounds
from scipy.interpolate import griddata
from shapely.geometry import LineString, box, mapping

from src.kruinlijn.profiel import extract_kniklijnen


def laz_to_dtm(laz_path, output_path, bbox=None, resolution=0.25, class_filter=None):
    """Maak een DTM van AHN4 puntenwolk (LAZ).

    Parameters
    ----------
    laz_path : str
        Pad naar LAZ bestand
    output_path : str
        Pad voor output GeoTIFF
    bbox : tuple (xmin, ymin, xmax, ymax) of None
        Clip naar bounding box (RD New)
    resolution : float
        Pixelgrootte in meters (default 0.25m)
    class_filter : list of int or None
        LAS classificaties om te gebruiken. None = alleen grond (2).
        AHN4: 1=unclassified, 2=ground, 6=building, 9=water, 26=high-voltage
    """
    if class_filter is None:
        class_filter = [2]  # Alleen grondpunten

    print(f"Laden puntenwolk: {laz_path}")
    las = laspy.read(laz_path)
    print(f"  Totaal punten: {len(las.points):,}")

    # Classificaties tonen
    classes, counts = np.unique(las.classification, return_counts=True)
    for c, n in zip(classes, counts):
        label = {0: "never classified", 1: "unclassified", 2: "ground",
                 6: "building", 9: "water", 26: "high-voltage"}.get(int(c), "other")
        marker = " <-- " if int(c) in class_filter else ""
        print(f"  Klasse {c:3d} ({label}): {n:>10,}{marker}")

    # Filter op classificatie
    mask = np.isin(las.classification, class_filter)
    x = np.array(las.x[mask])
    y = np.array(las.y[mask])
    z = np.array(las.z[mask])
    print(f"  Na filter: {len(x):,} grondpunten")

    # Clip op bbox
    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        clip = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        x, y, z = x[clip], y[clip], z[clip]
        print(f"  Na clip: {len(x):,} punten in bbox")
    else:
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()

    if len(x) < 100:
        print("  Te weinig punten!")
        return None

    # Grid maken
    cols = int(np.ceil((xmax - xmin) / resolution))
    rows = int(np.ceil((ymax - ymin) / resolution))
    print(f"  Grid: {cols}x{rows} pixels ({resolution}m)")

    # Interpoleren naar grid via griddata (linear)
    xi = np.linspace(xmin + resolution / 2, xmax - resolution / 2, cols)
    yi = np.linspace(ymax - resolution / 2, ymin + resolution / 2, rows)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    print("  Interpoleren...")
    grid = griddata(
        (x, y), z,
        (xi_grid, yi_grid),
        method="linear",
        fill_value=np.nan,
    )

    # Vul kleine gaten met nearest-neighbor
    nan_mask = np.isnan(grid)
    if nan_mask.any():
        grid_nn = griddata(
            (x, y), z,
            (xi_grid[nan_mask], yi_grid[nan_mask]),
            method="nearest",
        )
        grid[nan_mask] = grid_nn

    grid = grid.astype(np.float32)

    # Schrijf als GeoTIFF
    transform = from_bounds(xmin, ymin, xmax, ymax, cols, rows)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype="float32",
        crs="EPSG:28992",
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(grid, 1)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Geschreven: {output_path} ({size_mb:.1f} MB)")
    return output_path


def run_comparison(laz_path, traject_geojson, dtm_50cm_path=None, output_dir="output/pointcloud_test"):
    """Vergelijk kniklijnen uit 25cm puntenwolk-DTM vs 50cm standaard DTM."""
    os.makedirs(output_dir, exist_ok=True)

    # Laad traject
    gdf = gpd.read_file(traject_geojson)
    if gdf.crs and gdf.crs.to_epsg() != 28992:
        gdf = gdf.to_crs(epsg=28992)
    trajectory = gdf.geometry.iloc[0]

    # Zoek een stuk traject dat volledig binnen LAZ dekking valt
    # LAZ subtile 39GN1_20: x=163980-165020
    # Neem punten met x < 165000 (ruim binnen dekking)
    segment_pts = []
    for d in np.arange(0, trajectory.length, 1.0):
        p = trajectory.interpolate(d)
        if 164050 < p.x < 164950:  # Ruim binnen LAZ extent
            segment_pts.append((p.x, p.y))
    if len(segment_pts) < 50:
        print("Te weinig punten binnen LAZ dekking!")
        return
    segment = LineString(segment_pts)
    print(f"Segment binnen LAZ dekking: {segment.length:.0f}m, {len(segment_pts)} punten")

    # Bbox rondom segment met 150m buffer
    bounds = segment.bounds
    bbox = (bounds[0] - 150, bounds[1] - 150, bounds[2] + 150, bounds[3] + 150)
    print(f"Test segment: {segment.length:.0f}m, bbox: {bbox}")

    # 1. Maak 25cm DTM uit puntenwolk
    dtm_25cm_path = os.path.join(output_dir, "dtm_25cm_pointcloud.tif")
    laz_to_dtm(laz_path, dtm_25cm_path, bbox=bbox, resolution=0.25)

    # 2. Maak ook 50cm DTM uit puntenwolk (voor vergelijking)
    dtm_50cm_pc_path = os.path.join(output_dir, "dtm_50cm_pointcloud.tif")
    laz_to_dtm(laz_path, dtm_50cm_pc_path, bbox=bbox, resolution=0.50)

    # 3. Draai knikpunt-detectie op alle DTMs
    results = {}
    dtm_configs = [
        ("PC 25cm", dtm_25cm_path),
        ("PC 50cm", dtm_50cm_pc_path),
    ]
    if dtm_50cm_path and os.path.exists(dtm_50cm_path):
        dtm_configs.append(("AHN4 server 50cm", dtm_50cm_path))

    for label, dtm_path in dtm_configs:
        print(f"\n=== Knikpunt-detectie: {label} ===")
        lines, points, crs = extract_kniklijnen(
            segment,
            dtm_path,
            profile_spacing=1.0,
            half_width=120.0,
            smooth_sigma=3.0,
            line_smooth_sigma=10.0,
        )
        results[label] = (lines, points, crs)

    # 4. Schrijf resultaten naar GeoPackage
    gpkg_path = os.path.join(output_dir, "vergelijking_pointcloud.gpkg")
    all_features = []
    for label, (lines, points, crs) in results.items():
        for name, line in lines.items():
            if line is not None:
                all_features.append({
                    "geometry": line,
                    "bron": label,
                    "naam": name,
                    "lengte_m": line.length,
                })

    if all_features:
        gdf_out = gpd.GeoDataFrame(all_features, crs="EPSG:28992")
        gdf_out.to_file(gpkg_path, layer="lijnen", driver="GPKG")
        print(f"\nResultaten: {gpkg_path}")

    # 5. Vergelijk lijnposities
    print("\n=== Vergelijking ===")
    baseline_label = dtm_configs[0][0]
    for label, (lines, _, _) in results.items():
        if label == baseline_label:
            continue
        print(f"\n  {baseline_label} vs {label}:")
        base_lines = results[baseline_label][0]
        for name in ["binnenkruinlijn", "buitenkruinlijn", "binnenteenlijn", "buitenteenlijn"]:
            l1 = base_lines.get(name)
            l2 = lines.get(name)
            if l1 is not None and l2 is not None:
                # Hausdorff-afstand
                hausdorff = l1.hausdorff_distance(l2)
                # Gemiddelde afstand (sample punten)
                dists = []
                for d in np.linspace(0, l1.length, 100):
                    p = l1.interpolate(d)
                    dists.append(l2.distance(p))
                mean_dist = np.mean(dists)
                print(f"    {name}: gem={mean_dist:.2f}m, hausdorff={hausdorff:.2f}m")
            else:
                print(f"    {name}: niet beschikbaar in beide")

    # 6. Schrijf punten per bron
    for label, (lines, points, crs) in results.items():
        safe_label = label.replace(" ", "_").replace("/", "_")
        for name, pts_list in points.items():
            if pts_list:
                pts_gdf = gpd.GeoDataFrame(
                    [{"geometry": gpd.points_from_xy([p[0]], [p[1]])[0],
                      "hoogte": p[2], "naam": name, "bron": label}
                     for p in pts_list],
                    crs="EPSG:28992",
                )
                pts_gdf.to_file(gpkg_path, layer=f"punten_{safe_label}", driver="GPKG",
                                mode="a" if os.path.exists(gpkg_path) else "w")

    print(f"\nAlles geschreven naar: {gpkg_path}")
    return results


if __name__ == "__main__":
    laz_path = "data/pointcloud/39GN1_20.LAZ"
    traject_path = "data/raw/zwo_sample_2km.geojson"
    dtm_50cm = "output/profiel_zwo/dtm_t1_00000.tif"  # Bestaande 50cm DTM

    if not os.path.exists(laz_path):
        print(f"LAZ bestand niet gevonden: {laz_path}")
        print("Download eerst: https://geotiles.citg.tudelft.nl/AHN4_T/39GN1_20.LAZ")
        sys.exit(1)

    run_comparison(laz_path, traject_path, dtm_50cm)
