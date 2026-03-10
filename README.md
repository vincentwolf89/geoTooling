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

## Installatie

```bash
git clone <repo-url> && cd geoTooling
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux/Mac

pip install -e ".[dl]"
```

## Gebruik

### Profiel-analyse op dijktrajecten (primaire aanpak)

```bash
# Alle 3 ZWO trajecten (West Maas en Waal, Beuningen, Druten)
python examples/run_profiel_zwo.py

# Enkel traject testen
python examples/test_profiel_v4.py
```

### Output

Per traject een GeoPackage met:
- **Laag `lijnen`**: 4 kniklijnen (binnenkruin, buitenkruin, binnenteen, buitenteen)
- **Laag `punten`**: ~10.000 raw knikpunten per lijn met hoogte

### Nauwkeurigheid (mediaan afstand tot WSRL PVVR referentie)

| Lijn | Traject 1 | Traject 2 | Traject 3 |
|------|-----------|-----------|-----------|
| Binnenkruinlijn | 3.9m | 3.1m | 2.6m |
| Buitenkruinlijn | 2.5m | 2.2m | 2.0m |
| Binnenteenlijn | 7.6m | 13.9m | 7.4m |
| Buitenteenlijn | 6.2m | 2.3m | 4.1m |

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
