---
allowedTools:
  - Edit
  - Write
  - Read
  - Glob
  - Grep
  - Bash(*)
  - TodoWrite
  - Agent
  - WebFetch
  - WebSearch
  - NotebookEdit
---

# Project: geoTooling - Dijksegmentatie via Deep Learning

## Structuur
- `src/kruinlijn/dl/` — Attention U-Net + ASPP + SE model, training, predict, vectorize
- `src/kruinlijn/data.py` — Data downloads (AHN4 DTM/DSM, luchtfoto, BGT) + DTB loader
- `src/kruinlijn/pipeline.py` — Label-generatie uit referentielijnen
- `examples/train_with_dtb.py` — Training met DTB referentielijnen (RWS)
- `examples/train_with_real_labels.py` — Training met WSRL referentielijnen
- `examples/predict_area.py` — Voorspelling op nieuw dijkgebied
- `data/raw/*.geojson` — Referentielijnen, DTB-export (EPSG:28992), trajecten

## Conventies
- CRS: EPSG:28992 (RD New) voor rasters en output, EPSG:4326 voor referentielijnen
- 10 klassen: 0=achtergrond, 1=kruin, 2=talud_binnen, 3=binnenberm, 4=teen_binnen, 5=talud_buiten, 6=buitenberm, 7=teen_buiten, 8=insteek, 9=sloot
- Model: 9 kanalen (DTM + slope + aspect + curvature + TPI + nDSM + R + G + B)
- Taal: code in het Engels, comments/prints in het Nederlands

## Data bronnen
- AHN4 DTM: https://ahn.arcgisonline.nl/arcgis/rest/services/Hoogtebestand/AHN4_DTM_50cm/ImageServer
- AHN4 DSM: https://ahn.arcgisonline.nl/arcgis/rest/services/Hoogtebestand/AHN4_DSM_50cm/ImageServer
- PDOK luchtfoto: https://services.arcgisonline.nl/arcgis/rest/services/Luchtfoto/Luchtfoto/MapServer/export
- BGT (waterdeel/sloot): https://api.pdok.nl/lv/bgt/ogc/v1_0
- DTB (dijklijnen RWS): GeoJSON export, omschr: kruinlijn, talud(bovenkant), talud(onderkant)

## Permissions
- Bash commands mogen standaard draaien zonder bevestiging (incl. API calls, downloads, git)
- Python scripts en API calls mogen altijd zonder bevestiging
- Alleen destructieve git operaties (force push, reset --hard) vragen om bevestiging
