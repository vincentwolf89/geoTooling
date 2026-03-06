# geoTooling — Kruinlijndetectie

Automatische detectie van kruinlijnen (en andere dijkonderdelen) op basis van DTM-data, via twee benaderingen:

1. **Morfologische analyse** — dwarsprofielen langs de dijk, knikpuntdetectie via kromming/2e afgeleide
2. **Deep Learning** — U-Net semantic segmentation voor per-pixel classificatie (kruin, talud, teen)
3. **Hybride pipeline** — morfologische methode genereert automatisch trainingsdata voor het DL-model

## Projectstructuur

```
geoTooling/
├── src/kruinlijn/
│   ├── morpho/           # Klassieke morfologische analyse
│   │   ├── profiles.py   # Dwarsprofielgeneratie langs dijklijn
│   │   └── crest.py      # Kruinpuntdetectie (peak/curvature)
│   ├── dl/               # Deep learning segmentatie
│   │   ├── dataset.py    # PyTorch dataset (DTM tiles + labels)
│   │   ├── model.py      # U-Net architectuur
│   │   ├── train.py      # Training loop
│   │   └── predict.py    # Sliding window voorspelling
│   └── pipeline.py       # Hybride pipeline (morpho -> labels -> DL)
├── examples/
│   ├── demo_morpho.py    # Demo: alleen morfologische analyse
│   └── demo_hybrid.py    # Demo: volledige hybride pipeline
├── data/                 # Data (niet in git)
├── models/checkpoints/   # Model checkpoints (niet in git)
└── pyproject.toml
```

## Installatie

```bash
# Clone en maak virtual environment
git clone <repo-url> && cd geoTooling
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Basis (alleen morfologische analyse)
pip install -e .

# Met deep learning support
pip install -e ".[dl]"

# Development
pip install -e ".[dev]"
```

## Gebruik

### Morfologische analyse

```python
from shapely.geometry import LineString
from kruinlijn.pipeline import morpho_pipeline

# Definieer een hartlijn van de dijk (of lees uit GeoPackage)
centerline = LineString([(x1, y1), (x2, y2), ...])

# Detecteer kruinlijn
crest_line, points_gdf = morpho_pipeline(
    dtm_path="data/raw/dtm.tif",
    centerline=centerline,
    output_gpkg="output/kruinlijn.gpkg",
    spacing=5.0,       # 5m tussen profielen
    width=40.0,        # 40m breed dwarsprofiel
    method="curvature", # of "peak"
)
```

### Hybride pipeline (morfologisch + DL)

```bash
python examples/demo_hybrid.py data/raw/dtm.tif data/raw/hartlijn.gpkg output/
```

Dit doet automatisch:
1. Kruinlijndetectie via morfologische analyse
2. Genereren van segmentatielabels (kruin/talud/teen zones)
3. Knippen van trainingstiles
4. Trainen van een U-Net model
5. Voorspelling op het volledige DTM

### Klassen (segmentatie)

| Label | Klasse         |
|-------|----------------|
| 0     | Achtergrond    |
| 1     | Kruin          |
| 2     | Talud binnen   |
| 3     | Teen binnen    |
| 4     | Talud buiten   |
| 5     | Teen buiten    |

## Benodigde input

- **DTM**: GeoTIFF, bij voorkeur 0.5m of 1m resolutie (AHN3/AHN4)
- **Hartlijn**: GeoPackage/Shapefile met een LineString die globaal over de dijk loopt
- Optioneel: luchtfoto (nog niet geimplementeerd als extra inputkanaal)

## Methode-details

### Morfologisch
- Genereert dwarsprofielen loodrecht op de dijkhartlijn
- Per profiel: Gaussische smoothing -> 2e afgeleide (kromming) -> piekdetectie
- Kruinpunten worden verbonden tot een vloeiende kruinlijn

### Deep Learning
- Lichtgewicht U-Net (32 base features, 4 encoder-blokken)
- Input: DTM + slope (2 kanalen), optioneel aspect
- Class weighting: kruin-klasse krijgt extra gewicht (3x)
- Sliding window met overlap voor naadloze voorspelling
