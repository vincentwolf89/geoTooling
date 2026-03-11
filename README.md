# geoTooling — Automatische kniklijnen-extractie uit AHN

Automatische detectie van dijkkniklijnen (kruinranden, teenlijnen) uit AHN4 DTM hoogte-data via dwarsprofiel-analyse. Produceert 4 continue lijnen per dijktraject: binnenkruinlijn, buitenkruinlijn, binnenteenlijn, buitenteenlijn.

## Aanpak: Profiel-analyse

1. **Dwarsprofielen** — Elke meter een loodrecht profiel (240m breed) op het AHN4 DTM
2. **Knikpunt-detectie** — Kruinranden via krommingsanalyse (2e afgeleide), teenlijnen via helling-wandeling
3. **Kwaliteitsfilter** — Slechte profielen (bebouwing, opritten) worden afgewezen op basis van prominentie, ruwheid en hoogtevariatie
4. **AHN-snap** — Elk knikpunt wordt gesnapt naar het dichtstbijzijnde terrein-kenmerk op de DTM (3m zoekradius)
5. **Smoothing** — Gaussian smoothing op XY-coordinaten voor vloeiende lijnen
6. **Output** — GeoPackage met lijnen + raw knikpunten (incl. hoogte)

## Projectstructuur

```
geoTooling/
├── src/kruinlijn/
│   ├── profiel.py          # Knikpunt-detectie via dwarsprofiel-analyse
│   ├── data.py             # Data downloads (AHN4 DTM/DSM, luchtfoto, BGT)
│   ├── pipeline.py         # Label-generatie uit referentielijnen
│   └── dl/                 # Deep Learning aanpak (experimenteel)
│       ├── model.py        # Attention U-Net + ASPP + SE blocks
│       ├── train.py        # Training pipeline
│       ├── predict.py      # Sliding window voorspelling
│       └── vectorize.py    # Pixel → vectorlijnen
├── examples/
│   ├── run_profiel_zwo.py        # Volledige pipeline: 3 ZWO trajecten
│   ├── test_profiel_v4.py        # Test op enkel traject + validatie
│   ├── predict_area.py           # DL-voorspelling op nieuw dijkgebied
│   └── train_wsrl.py             # DL-training met WSRL labels
├── data/raw/               # Referentielijnen, trajecten, PVVR (niet in git)
└── pyproject.toml
```

## Snelstart: kniklijnen genereren voor eigen dijktraject

### 1. Installatie

```bash
git clone <repo-url> && cd geoTooling
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux/Mac
pip install -e ".[dl]"
```

### 2. Traject-bestand voorbereiden

Maak een GeoJSON met één of meer dijktrajecten als LineString in **EPSG:28992** (RD New):

```json
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "properties": {"naam": "Mijn dijktraject"},
    "geometry": {"type": "LineString", "coordinates": [[155000, 430000], [156000, 430500]]}
  }]
}
```

De lijn moet het **midden van de dijkkruin** volgen (hartlijn/aslijn). Lengte maakt niet uit — de DTM wordt automatisch in secties gedownload.

### 3. Script draaien

```python
from pathlib import Path
import os
os.environ.pop("PROJ_LIB", None)  # voorkom PROJ conflict

import geopandas as gpd
import rasterio
import numpy as np
from rasterio.merge import merge
from shapely.geometry import LineString, Point

from kruinlijn.data import download_ahn4_dtm
from kruinlijn.profiel import extract_kniklijnen

# --- Config ---
TRAJECT_FILE = "pad/naar/mijn_traject.geojson"
OUTPUT_DIR = Path("output/mijn_resultaat")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Lees traject
trajecten = gpd.read_file(TRAJECT_FILE)

for idx, row in trajecten.iterrows():
    geom = row.geometry
    if geom.has_z:
        geom = LineString([(x, y) for x, y, *_ in geom.coords])

    # Download DTM in secties van 2km
    section_files = []
    for start in range(0, int(geom.length) + 1, 2000):
        end = min(start + 2000, geom.length)
        pts = np.array([(geom.interpolate(d).x, geom.interpolate(d).y)
                        for d in np.arange(start, end, 10)])
        bbox = (*pts.min(axis=0) - 200, *pts.max(axis=0) + 200)
        section_path = OUTPUT_DIR / f"dtm_{idx}_{start:05d}.tif"
        if not section_path.exists():
            download_ahn4_dtm(bbox, section_path)
        section_files.append(section_path)

    # Merge DTM secties
    dtm_path = OUTPUT_DIR / f"dtm_{idx}_merged.tif"
    if not dtm_path.exists():
        datasets = [rasterio.open(f) for f in section_files]
        mosaic, out_transform = merge(datasets)
        for ds in datasets:
            ds.close()
        profile = rasterio.open(section_files[0]).profile.copy()
        profile.update(width=mosaic.shape[2], height=mosaic.shape[1], transform=out_transform)
        with rasterio.open(dtm_path, "w", **profile) as dst:
            dst.write(mosaic)

    # Extraheer kniklijnen
    lines, points, crs = extract_kniklijnen(
        trajectory=geom,
        dtm_path=str(dtm_path),
        profile_spacing=1.0,     # profiel elke meter
        half_width=120.0,        # 120m naar elke kant
        smooth_sigma=3.0,        # smoothing van profielen
        line_smooth_sigma=10.0,  # smoothing van output-lijnen
    )

    # Opslaan als GeoPackage
    output_gpkg = OUTPUT_DIR / f"kniklijnen_{idx}.gpkg"

    rows_out = []
    for naam, line_geom in lines.items():
        if line_geom is not None:
            rows_out.append({"naam": naam, "lengte_m": round(line_geom.length, 1), "geometry": line_geom})
    if rows_out:
        gpd.GeoDataFrame(rows_out, crs=crs).to_file(output_gpkg, driver="GPKG", layer="lijnen")

    pt_rows = []
    for naam, pts in points.items():
        for x, y, z in pts:
            pt_rows.append({"naam": naam, "hoogte": round(z, 2), "geometry": Point(x, y)})
    if pt_rows:
        gpd.GeoDataFrame(pt_rows, crs=crs).to_file(output_gpkg, driver="GPKG", layer="punten")

    print(f"Klaar: {output_gpkg}")
```

Of gebruik het meegeleverde voorbeeld-script voor de 3 ZWO trajecten:

```bash
python examples/run_profiel_zwo.py
```

### 4. Parameters afstemmen

| Parameter | Default | Beschrijving |
|-----------|---------|-------------|
| `profile_spacing` | 1.0 | Afstand tussen dwarsprofielen (m). Lager = fijner maar trager |
| `half_width` | 120.0 | Halve breedte van dwarsprofiel (m). Vergroot bij brede uiterwaarden |
| `smooth_sigma` | 3.0 | Smoothing van individuele hoogte-profielen |
| `line_smooth_sigma` | 10.0 | Smoothing van output-lijnen. Hoger = vloeiender, lager = meer terreindetail |

### Output

Per traject een GeoPackage met:
- **Laag `lijnen`**: 4 kniklijnen (binnenkruin, buitenkruin, binnenteen, buitenteen) met vertex elke ~1m
- **Laag `punten`**: raw knikpunten per lijn met hoogte (voor inspectie/handmatige correctie)

### Nauwkeurigheid (mediaan afstand tot WSRL PVVR referentie)

| Lijn | Traject 1 | Traject 2 | Traject 3 |
|------|-----------|-----------|-----------|
| Binnenkruinlijn | 3.7m | 2.9m | 2.7m |
| Buitenkruinlijn | 2.4m | 2.1m | 2.0m |
| Binnenteenlijn | 7.4m | 14.0m | 7.5m |
| Buitenteenlijn | 6.2m | 2.2m | 4.2m |

## Algoritme details

### Knikpunt-detectie per profiel

- **Kruinrand**: Sterkste negatieve kromming (convexe knik) binnen 15m van kruintop
- **Teenlijnen**: Wandel talud af vanaf kruinrand, zoek waar helling < 0.05 en hoogte < kruin - 1.5m
  - Buitenzijde: eerste vlakke plek (geen bermen)
  - Binnenzijde: laagste vlakke plek (passeert bermen)
- **Buiten-kant detectie**: Gewogen stemming op taludsteilheid (2x) + hoogte ver weg

### Validatie criteria per profiel

- Minimaal 2m kruinbreedte
- Minimaal 5m afstand kruin-teen
- Minimaal 1.5m hoogteverschil kruin-teen
- Correcte volgorde: buitenteen < buitenkruin < binnenkruin < binnenteen

## Data bronnen

| Bron | Beschrijving | URL |
|------|-------------|-----|
| AHN4 DTM | Hoogtemodel (maaiveldhoogte, 50cm) | ArcGIS ImageServer |
| WSRL PVVR | Referentielijnen waterschap | portal.wsrl.nl |
| DTB | Professioneel ingemeten dijklijnen (RWS) | GeoJSON export |
