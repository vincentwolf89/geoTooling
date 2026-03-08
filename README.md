# geoTooling — Dijksegmentatie via Deep Learning

Automatische detectie van dijkonderdelen (kruinlijn, teenlijnen, taluds, bermen, insteek, sloot) op basis van AHN4 hoogte-data en luchtfoto's, met een Attention U-Net segmentatiemodel.

## Aanpak

1. **Referentielijnen als labels** — DTB-lijnen (RWS) of handmatig ingemeten lijnen worden samen met BGT waterdelen omgezet naar segmentatie-rasters
2. **Multi-channel input** — 9 kanalen: DTM + slope + aspect + curvature + TPI + nDSM + R + G + B
3. **Attention U-Net + ASPP + SE** — Segmentatiemodel met attention gates, atrous spatial pyramid pooling, en squeeze-excitation channel attention
4. **Sliding window predict** — Voorspelling op willekeurig grote gebieden via overlappende tiles
5. **Vectorisatie** — Pixel-classificatie wordt omgezet naar vectorlijnen (GeoPackage)

## Projectstructuur

```
geoTooling/
├── src/kruinlijn/
│   ├── data.py             # Data downloads (AHN4 DTM/DSM, luchtfoto, BGT, DTB)
│   ├── pipeline.py         # Label-generatie uit referentielijnen
│   └── dl/
│       ├── dataset.py      # PyTorch dataset (9 kanalen, augmentatie)
│       ├── model.py        # Attention U-Net + ASPP + SE blocks
│       ├── train.py        # Training: Focal Loss + Dice Loss, deep supervision
│       ├── predict.py      # Sliding window voorspelling + post-processing
│       └── vectorize.py    # Pixel → vectorlijnen
├── examples/
│   ├── train_with_dtb.py         # Training met DTB referentielijnen
│   ├── train_with_real_labels.py # Training met WSRL referentielijnen
│   └── predict_area.py           # Voorspelling op nieuw dijkgebied
├── data/raw/               # Referentielijnen, trajecten (niet in git)
└── pyproject.toml
```

## Installatie

```bash
git clone <repo-url> && cd geoTooling
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux/Mac

pip install -e ".[dl]"
```

## Gebruik

### Training met DTB referentielijnen

```bash
# Standaard: gebruikt data/raw/dtb_kruinlijnen_selectie.geojson
python examples/train_with_dtb.py

# Of met een eigen DTB-export:
python examples/train_with_dtb.py data/raw/mijn_dtb.geojson
```

### Voorspelling op een nieuw gebied

```bash
python examples/predict_area.py data/raw/traject.geojson [model.pt] [output_dir]
```

### Klassen (10-klasse segmentatie)

| Label | Klasse         | Beschrijving                     |
|-------|----------------|----------------------------------|
| 0     | Achtergrond    | Geen dijk                        |
| 1     | Kruin          | Kruinzone van de dijk            |
| 2     | Talud binnen   | Binnenzijde (polder) talud       |
| 3     | Binnenberm     | Berm aan polderzijde             |
| 4     | Teen binnen    | Teen aan polderzijde             |
| 5     | Talud buiten   | Buitenzijde (water) talud        |
| 6     | Buitenberm     | Berm aan waterzijde              |
| 7     | Teen buiten    | Teen aan waterzijde              |
| 8     | Insteek        | Rand waterdeel (overgang sloot)   |
| 9     | Sloot          | Wateroppervlak                   |

## Data bronnen

| Bron | Beschrijving | URL |
|------|-------------|-----|
| AHN4 DTM | Hoogtemodel (maaiveldhoogte, 50cm) | ArcGIS ImageServer |
| AHN4 DSM | Oppervlaktemodel (incl. objecten) | ArcGIS ImageServer |
| PDOK Luchtfoto | RGB luchtfoto | ArcGIS MapServer |
| BGT Waterdeel | Sloten, waterlopen | PDOK OGC Features API |
| DTB | Professioneel ingemeten dijklijnen (RWS) | WFS / GeoJSON export |

## Model architectuur

- **Backbone**: Attention U-Net (32 base features, 4 encoder blokken)
- **Bottleneck**: ASPP (Atrous Spatial Pyramid Pooling) met dilations 6, 12 + global pooling
- **Channel attention**: Squeeze-Excitation blocks per ConvBlock
- **Residual connections**: 1x1 shortcut conv bij kanaalwijziging
- **Deep supervision**: Auxiliary outputs op decoder level 2 en 3
- **Loss**: Focal Loss (γ=2) + Dice Loss met automatische class weights
- **~14.7M parameters**, 9 input kanalen, 10 output klassen

## Input kanalen

| # | Kanaal | Beschrijving |
|---|--------|-------------|
| 1 | DTM | Maaiveldhoogte (genormaliseerd) |
| 2 | Slope | Helling (1e afgeleide) |
| 3 | Aspect | Hellingsrichting |
| 4 | Curvature | Kromming (Laplaciaan) |
| 5 | TPI | Topographic Position Index (hoogte t.o.v. omgeving) |
| 6 | nDSM | DSM - DTM (vegetatie/objecthoogte) |
| 7 | R | Rood kanaal luchtfoto |
| 8 | G | Groen kanaal luchtfoto |
| 9 | B | Blauw kanaal luchtfoto |
