"""Profiel-analyse op alle ZWO trajecten + validatie tegen PVVR.

Download DTM per sectie, merge, extraheer kniklijnen, valideer.
"""

from pathlib import Path
import sys
import os

os.environ.pop("PROJ_LIB", None)

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from kruinlijn.data import download_ahn4_dtm
from kruinlijn.profiel import extract_kniklijnen

# --- Config ---
TRAJECT_PATH = Path("data/raw/Trajecten ZWO.geojson")
PVVR_DIR = Path("data/raw/wsrl_pvvr")
OUTPUT_DIR = Path("output/profiel_zwo")
SECTION_LENGTH = 2000  # DTM download per 2km
DTM_BUFFER = 200  # buffer rond traject voor DTM

PVVR_FILES = {
    "binnenkruinlijn": "wsrl_pvvr_binnenkruinlijn.geojson",
    "buitenkruinlijn": "wsrl_pvvr_buitenkruinlijn.geojson",
    "binnenteenlijn": "wsrl_pvvr_binnenteenlijn.geojson",
    "buitenteenlijn": "wsrl_pvvr_buitenteenlijn.geojson",
}


def download_dtm_for_trajectory(trajectory, traject_id, output_dir):
    """Download AHN4 DTM secties en merge tot een bestand."""
    merged_path = output_dir / f"dtm_t{traject_id}_merged.tif"
    if merged_path.exists():
        print(f"  DTM bestaat al: {merged_path}")
        return merged_path

    if trajectory.has_z:
        trajectory = LineString([(x, y) for x, y, *_ in trajectory.coords])

    total = trajectory.length
    section_files = []

    for start in range(0, int(total) + 1, SECTION_LENGTH):
        section_path = output_dir / f"dtm_t{traject_id}_{start:05d}.tif"

        if section_path.exists():
            section_files.append(section_path)
            continue

        end = min(start + SECTION_LENGTH, total)
        # Bbox van dit stuk traject + buffer
        pts = []
        for d in np.arange(start, end, 10):
            pt = trajectory.interpolate(d)
            pts.append((pt.x, pt.y))
        pts = np.array(pts)
        minx, miny = pts.min(axis=0) - DTM_BUFFER
        maxx, maxy = pts.max(axis=0) + DTM_BUFFER

        print(f"  Downloaden DTM sectie {start}-{int(end)}m...")
        try:
            download_ahn4_dtm((minx, miny, maxx, maxy), section_path)
            section_files.append(section_path)
        except Exception as e:
            print(f"  FOUT bij download: {e}")

    if not section_files:
        return None

    # Merge
    print(f"  Mergen {len(section_files)} secties...")
    datasets = [rasterio.open(f) for f in section_files]
    mosaic, out_transform = merge(datasets)
    for ds in datasets:
        ds.close()

    profile = rasterio.open(section_files[0]).profile.copy()
    profile.update(
        width=mosaic.shape[2],
        height=mosaic.shape[1],
        transform=out_transform,
    )
    with rasterio.open(merged_path, "w", **profile) as dst:
        dst.write(mosaic)

    print(f"  DTM opgeslagen: {merged_path}")
    return merged_path


def validate_against_pvvr(lines, trajectory, traject_id):
    """Valideer kniklijnen tegen PVVR referentie."""
    if trajectory.has_z:
        trajectory = LineString([(x, y) for x, y, *_ in trajectory.coords])
    traject_buffer = trajectory.buffer(200)

    print(f"\n  --- Validatie traject {traject_id} ---")

    # Kruisingen
    print("  Kruisingen:")
    line_names = [n for n in lines if lines[n] is not None]
    has_crossing = False
    for i, a in enumerate(line_names):
        for b in line_names[i + 1:]:
            if lines[a].crosses(lines[b]):
                print(f"    FOUT: {a} x {b}")
                has_crossing = True
    if not has_crossing:
        print("    OK: geen kruisingen")

    # Hoogte
    print("  Hoogte:")
    dtm_path = OUTPUT_DIR / f"dtm_t{traject_id}_merged.tif"
    with rasterio.open(dtm_path) as src:
        dtm = src.read(1).astype(np.float32)
        transform = src.transform

    def sample_height(geom, n=100):
        heights = []
        for i in range(n):
            pt = geom.interpolate(i / (n - 1), normalized=True)
            inv = ~transform
            col, row = inv * (pt.x, pt.y)
            col, row = int(round(col)), int(round(row))
            h, w = dtm.shape
            if 0 <= row < h and 0 <= col < w:
                val = dtm[row, col]
                if -10 < val < 100:
                    heights.append(val)
        return np.array(heights) if heights else np.array([np.nan])

    for naam, geom in lines.items():
        if geom is None:
            continue
        h = sample_height(geom)
        print(f"    {naam}: mediaan={np.nanmedian(h):.2f}m")

    # PVVR vergelijking
    print("  PVVR vergelijking:")
    results = {}
    for naam, pvvr_file in PVVR_FILES.items():
        our_line = lines.get(naam)
        if our_line is None:
            print(f"    {naam}: ONTBREEKT")
            continue

        pvvr_path = PVVR_DIR / pvvr_file
        if not pvvr_path.exists():
            continue
        pvvr = gpd.read_file(pvvr_path)
        pvvr_clipped = pvvr[pvvr.geometry.intersects(traject_buffer)]
        if len(pvvr_clipped) == 0:
            print(f"    {naam}: geen PVVR in gebied")
            continue

        pvvr_geom = unary_union(pvvr_clipped.geometry)

        distances = []
        for i in range(200):
            pt = our_line.interpolate(i / 199, normalized=True)
            distances.append(pvvr_geom.distance(pt))
        distances = np.array(distances)
        med = np.median(distances)
        p90 = np.percentile(distances, 90)
        results[naam] = med
        print(f"    {naam}: mediaan={med:.1f}m  P90={p90:.1f}m")

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trajecten = gpd.read_file(TRAJECT_PATH)

    all_results = {}

    for idx, row in trajecten.iterrows():
        traject_id = idx + 1
        geom = row.geometry
        if geom.has_z:
            geom_2d = LineString([(x, y) for x, y, *_ in geom.coords])
        else:
            geom_2d = geom

        print(f"\n{'='*60}")
        print(f"Traject {traject_id}: {geom_2d.length:.0f}m")
        print(f"{'='*60}")

        # 1. DTM
        dtm_path = download_dtm_for_trajectory(geom_2d, traject_id, OUTPUT_DIR)
        if dtm_path is None:
            print("  FOUT: geen DTM beschikbaar")
            continue

        # 2. Profiel-analyse
        print("\n  Extractie kniklijnen...")
        lines, points, crs = extract_kniklijnen(
            trajectory=geom_2d,
            dtm_path=str(dtm_path),
            profile_spacing=1.0,
            half_width=120.0,
            smooth_sigma=3.0,
            line_smooth_sigma=10.0,
        )

        # 3. Opslaan: lijnen + punten in gpkg
        output_gpkg = OUTPUT_DIR / f"kniklijnen_t{traject_id}_v5.gpkg"
        rows = []
        for naam, line_geom in lines.items():
            if line_geom is not None:
                rows.append({
                    "naam": naam,
                    "type": "lijn",
                    "lengte_m": round(line_geom.length, 1),
                    "geometry": line_geom,
                })
        if rows:
            result_gdf = gpd.GeoDataFrame(rows, crs=crs)
            result_gdf.to_file(output_gpkg, driver="GPKG", layer="lijnen")

        # Punten apart opslaan
        pt_rows = []
        for naam, pts in points.items():
            for x, y, z in pts:
                pt_rows.append({
                    "naam": naam,
                    "hoogte": round(z, 2),
                    "geometry": Point(x, y),
                })
        if pt_rows:
            pts_gdf = gpd.GeoDataFrame(pt_rows, crs=crs)
            pts_gdf.to_file(output_gpkg, driver="GPKG", layer="punten")
            print(f"\n  Opgeslagen: {output_gpkg} ({len(pt_rows)} punten)")

        # 4. Validatie
        pvvr_results = validate_against_pvvr(lines, geom_2d, traject_id)
        all_results[traject_id] = pvvr_results

    # Samenvatting
    print(f"\n{'='*60}")
    print("SAMENVATTING")
    print(f"{'='*60}")
    for tid, results in all_results.items():
        print(f"\nTraject {tid}:")
        for naam, med in results.items():
            status = "OK" if med < 10 else "MATIG" if med < 20 else "SLECHT"
            print(f"  {naam}: {med:.1f}m [{status}]")


if __name__ == "__main__":
    main()
