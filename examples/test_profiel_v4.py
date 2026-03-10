"""Test profiel v4 (curvature-based) op ZWO traject 1 + zelf-validatie."""

from pathlib import Path
import sys
import os

# Fix PROJ_LIB conflict
os.environ.pop("PROJ_LIB", None)

import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from kruinlijn.profiel import extract_kniklijnen

# --- Paths ---
DTM_PATH = Path("output/profiel_zwo/dtm_t1_merged.tif")
TRAJECT_PATH = Path("data/raw/Trajecten ZWO.geojson")
PVVR_DIR = Path("data/raw/wsrl_pvvr")
OUTPUT_GPKG = Path("output/profiel_zwo/kniklijnen_t1_profiel_v4.gpkg")

# --- Load traject ---
trajecten = gpd.read_file(TRAJECT_PATH)
traject_geom = trajecten.geometry.iloc[0]  # West Maas en Waal
if traject_geom.has_z:
    traject_geom = LineString([(x, y) for x, y, *_ in traject_geom.coords])
print(f"Traject 1: {traject_geom.length:.0f}m")

# --- Extract kniklijnen ---
print("\n=== Extractie ===")
lines, crs = extract_kniklijnen(
    trajectory=traject_geom,
    dtm_path=str(DTM_PATH),
    profile_spacing=1.0,
    half_width=120.0,
    smooth_sigma=3.0,
    line_smooth_sigma=20.0,
)

# --- Save output ---
rows = []
for naam, geom in lines.items():
    if geom is not None:
        rows.append({"naam": naam, "lengte_m": round(geom.length, 1), "geometry": geom})
if rows:
    result = gpd.GeoDataFrame(rows, crs=crs)
    OUTPUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
    result.to_file(OUTPUT_GPKG, driver="GPKG")
    print(f"\nOpgeslagen: {OUTPUT_GPKG}")

# --- Zelf-validatie ---
print("\n=== VALIDATIE ===")

# 1. Check: geen kruisingen tussen lijnen
print("\n--- Kruisingen ---")
line_names = list(lines.keys())
for i, name_a in enumerate(line_names):
    for name_b in line_names[i+1:]:
        ga, gb = lines.get(name_a), lines.get(name_b)
        if ga is None or gb is None:
            continue
        if ga.crosses(gb):
            print(f"  FOUT: {name_a} kruist {name_b}!")
        else:
            print(f"  OK: {name_a} x {name_b} - geen kruising")

# 2. Check: hoogtevolgorde op sample punten
print("\n--- Hoogtevolgorde ---")
import rasterio
with rasterio.open(DTM_PATH) as src:
    dtm = src.read(1).astype(np.float32)
    transform = src.transform

def sample_height(geom, n=50):
    """Sample DTM hoogte op n punten langs een lijn."""
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
        print(f"  {naam}: ONTBREEKT")
        continue
    h = sample_height(geom)
    print(f"  {naam}: mediaan={np.nanmedian(h):.2f}m, range=[{np.nanmin(h):.2f}, {np.nanmax(h):.2f}]")

# Check: kruinlijnen hoger dan teenlijnen
bk_h = sample_height(lines.get("binnenkruinlijn")) if lines.get("binnenkruinlijn") else np.array([np.nan])
uk_h = sample_height(lines.get("buitenkruinlijn")) if lines.get("buitenkruinlijn") else np.array([np.nan])
bt_h = sample_height(lines.get("binnenteenlijn")) if lines.get("binnenteenlijn") else np.array([np.nan])
ut_h = sample_height(lines.get("buitenteenlijn")) if lines.get("buitenteenlijn") else np.array([np.nan])

kruin_med = np.nanmedian(np.concatenate([bk_h, uk_h]))
teen_med = np.nanmedian(np.concatenate([bt_h, ut_h]))
print(f"\n  Kruin mediaan: {kruin_med:.2f}m, Teen mediaan: {teen_med:.2f}m")
if kruin_med > teen_med:
    print(f"  OK: Kruin ({kruin_med:.2f}m) hoger dan teen ({teen_med:.2f}m)")
else:
    print(f"  FOUT: Teen ({teen_med:.2f}m) hoger dan kruin ({kruin_med:.2f}m)!")

# 3. Vergelijking met PVVR referentie
print("\n--- PVVR Referentie vergelijking ---")
pvvr_names = {
    "binnenkruinlijn": "wsrl_pvvr_binnenkruinlijn.geojson",
    "buitenkruinlijn": "wsrl_pvvr_buitenkruinlijn.geojson",
    "binnenteenlijn": "wsrl_pvvr_binnenteenlijn.geojson",
    "buitenteenlijn": "wsrl_pvvr_buitenteenlijn.geojson",
}

# Clip PVVR to traject area
traject_buffer = traject_geom.buffer(200)

for naam, pvvr_file in pvvr_names.items():
    our_line = lines.get(naam)
    if our_line is None:
        print(f"  {naam}: ONTBREEKT in output")
        continue

    pvvr_path = PVVR_DIR / pvvr_file
    pvvr = gpd.read_file(pvvr_path)

    # Filter PVVR to our area
    pvvr_clipped = pvvr[pvvr.geometry.intersects(traject_buffer)]
    if len(pvvr_clipped) == 0:
        print(f"  {naam}: geen PVVR referentie in gebied")
        continue

    # Merge PVVR to single line
    pvvr_geom = unary_union(pvvr_clipped.geometry)

    # Sample afstanden van onze lijn naar PVVR
    n_sample = 200
    distances = []
    for i in range(n_sample):
        pt = our_line.interpolate(i / (n_sample - 1), normalized=True)
        d = pvvr_geom.distance(pt)
        distances.append(d)

    distances = np.array(distances)
    med = np.median(distances)
    p90 = np.percentile(distances, 90)
    p95 = np.percentile(distances, 95)
    print(f"  {naam}: mediaan={med:.1f}m  P90={p90:.1f}m  P95={p95:.1f}m  (n={len(pvvr_clipped)} ref features)")

# 4. Diagnose binnenteen: sample offsets from trajectory
print("\n--- Binnenteen diagnose ---")
if lines.get("binnenteenlijn") is not None:
    bt_line = lines["binnenteenlijn"]
    pvvr_bt = gpd.read_file(PVVR_DIR / "wsrl_pvvr_binnenteenlijn.geojson")
    pvvr_bt_clipped = pvvr_bt[pvvr_bt.geometry.intersects(traject_buffer)]
    pvvr_bt_geom = unary_union(pvvr_bt_clipped.geometry)

    # Sample offset from trajectory for our line vs PVVR
    our_offsets = []
    pvvr_offsets = []
    for i in range(50):
        # Our line
        pt = bt_line.interpolate(i / 49, normalized=True)
        proj = traject_geom.project(Point(pt.x, pt.y))
        our_offsets.append(Point(pt.x, pt.y).distance(traject_geom.interpolate(proj)))

    # PVVR line
    for i in range(50):
        pt = pvvr_bt_geom.interpolate(i / 49, normalized=True)
        proj = traject_geom.project(Point(pt.x, pt.y))
        pvvr_offsets.append(Point(pt.x, pt.y).distance(traject_geom.interpolate(proj)))

    our_offsets = np.array(our_offsets)
    pvvr_offsets = np.array(pvvr_offsets)
    print(f"  Onze binnenteen offset van traject: mediaan={np.median(our_offsets):.1f}m, range=[{our_offsets.min():.1f}, {our_offsets.max():.1f}]")
    print(f"  PVVR binnenteen offset van traject: mediaan={np.median(pvvr_offsets):.1f}m, range=[{pvvr_offsets.min():.1f}, {pvvr_offsets.max():.1f}]")
    print(f"  -> Verschil: onze teen {np.median(our_offsets) - np.median(pvvr_offsets):.1f}m verder van traject")

print("\nKlaar!")
