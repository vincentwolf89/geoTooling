"""Download WSRL profielkniklijnen uit de legger (ArcGIS REST).

Downloadt alle kniklijnen per type en slaat op als GeoJSON in data/raw/.
Filtert op soort: Binnenkruinlijn, Buitenkruinlijn, Binnenteenlijn, Buitenteenlijn.

Gebruik:
    python examples/download_wsrl_legger.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests
from shapely.geometry import shape, mapping
from shapely.ops import linemerge

BASE_URL = (
    "https://portal.wsrl.nl/kaarten/rest/services/Waterveiligheid/"
    "Legger_en_Werkingsgebieden_Waterkeringen_Vastgesteld/MapServer/24"
)
QUERY_URL = BASE_URL + "/query"

# Lijntypes die we willen downloaden
TARGET_TYPES = [
    "Binnenkruinlijn",
    "Buitenkruinlijn",
    "Binnenteenlijn",
    "Buitenteenlijn",
    "Kniklijn (Binnen)",
    "Kniklijn (Buiten)",
]

OUTPUT_DIR = Path("data/raw/wsrl_legger")
MAX_RECORDS = 2000  # server limiet
MIN_SEGMENT_LENGTH = 50  # meters — filter heel korte fragmenten (waarschijnlijk fouten)


def fetch_page(soort: str, offset: int) -> list[dict]:
    """Haal één pagina features op voor een bepaald type."""
    params = {
        "where": f"WS_SOORTKNIKLIJN = '{soort}'",
        "outFields": "OBJECTID,WS_AFKORTINGKNIKLIJN,WS_SOORTKNIKLIJN,WS_HOOGTEKNIKLIJN",
        "outSR": "28992",
        "f": "geojson",
        "resultOffset": offset,
        "resultRecordCount": MAX_RECORDS,
    }
    r = requests.get(QUERY_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data.get("features", [])


def download_type(soort: str) -> list[dict]:
    """Download alle features van één type via paginering."""
    all_features = []
    offset = 0
    while True:
        features = fetch_page(soort, offset)
        if not features:
            break
        all_features.extend(features)
        print(f"    offset {offset}: {len(features)} features")
        if len(features) < MAX_RECORDS:
            break
        offset += MAX_RECORDS
        time.sleep(0.3)  # beleefd zijn naar de server
    return all_features


def filter_features(features: list[dict]) -> list[dict]:
    """Verwijder heel korte fragmenten (geometrische fouten)."""
    filtered = []
    for f in features:
        geom = shape(f["geometry"])
        if geom.length >= MIN_SEGMENT_LENGTH:
            filtered.append(f)
    return filtered


def save_geojson(features: list[dict], path: Path, soort: str) -> None:
    """Sla features op als GeoJSON (EPSG:28992)."""
    geojson = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::28992"}},
        "features": features,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(geojson, f)
    print(f"    Opgeslagen: {path} ({len(features)} features)")


def main():
    print("=" * 60)
    print("WSRL Legger — Profielkniklijnen downloaden")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    totals = {}

    for soort in TARGET_TYPES:
        safe_name = soort.lower().replace(" ", "_").replace("(", "").replace(")", "")
        out_path = OUTPUT_DIR / f"wsrl_legger_{safe_name}.geojson"

        if out_path.exists():
            print(f"\n[SKIP] {soort} — al aanwezig ({out_path.name})")
            with open(out_path) as f:
                existing = json.load(f)
            totals[soort] = len(existing["features"])
            continue

        print(f"\n[{soort}]")
        features = download_type(soort)
        print(f"  Totaal gedownload: {len(features)}")

        features = filter_features(features)
        print(f"  Na filter (>={MIN_SEGMENT_LENGTH}m): {len(features)}")

        save_geojson(features, out_path, soort)
        totals[soort] = len(features)

    print("\n" + "=" * 60)
    print("Samenvatting:")
    for soort, n in totals.items():
        print(f"  {soort}: {n} segmenten")
    total_length_km = "?"
    print(f"\nBestanden in: {OUTPUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
