"""Data-download functies: AHN4 DTM, luchtfoto, BGT waterdelen, DTB/HDSR referentielijnen."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import rasterio
import requests
from PIL import Image
from rasterio.transform import from_bounds
from shapely.geometry import shape, box, mapping

AHN4_URL = (
    "https://ahn.arcgisonline.nl/arcgis/rest/services"
    "/Hoogtebestand/AHN4_DTM_50cm/ImageServer"
)
AHN4_DSM_URL = (
    "https://ahn.arcgisonline.nl/arcgis/rest/services"
    "/Hoogtebestand/AHN4_DSM_50cm/ImageServer"
)
LUCHTFOTO_URL = (
    "https://services.arcgisonline.nl/arcgis/rest/services"
    "/Luchtfoto/Luchtfoto/MapServer/export"
)
BGT_OGC_URL = "https://api.pdok.nl/lv/bgt/ogc/v1_0"


def download_ahn4_dtm(
    bbox: tuple, output_path: Path, size_px: int = 2048
) -> tuple[int, int]:
    """Download AHN4 DTM via ArcGIS ImageServer.

    Parameters
    ----------
    bbox : tuple
        (minx, miny, maxx, maxy) in EPSG:28992.
    output_path : Path
        Uitvoerpad voor het GeoTIFF.
    size_px : int
        Maximale afmeting in pixels.

    Returns
    -------
    tuple[int, int]
        (breedte_px, hoogte_px)
    """
    minx, miny, maxx, maxy = bbox
    width = maxx - minx
    height = maxy - miny
    aspect = width / height

    if aspect >= 1:
        px_w = size_px
        px_h = max(64, int(size_px / aspect))
    else:
        px_h = size_px
        px_w = max(64, int(size_px * aspect))

    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "bboxSR": "28992",
        "imageSR": "28992",
        "size": f"{px_w},{px_h}",
        "format": "tiff",
        "pixelType": "F32",
        "noData": "-9999",
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }

    resp = requests.get(f"{AHN4_URL}/exportImage", params=params, timeout=120)
    resp.raise_for_status()

    if resp.headers.get("content-type", "").startswith("application/json"):
        raise RuntimeError(f"AHN4 server error: {resp.json()}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_suffix(".raw.tif")
    raw_path.write_bytes(resp.content)

    transform = from_bounds(minx, miny, maxx, maxy, px_w, px_h)
    with rasterio.open(raw_path) as src:
        data = src.read(1).astype(np.float32)

    data[data < -100] = np.nan

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=px_h,
        width=px_w,
        count=1,
        dtype="float32",
        crs="EPSG:28992",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(np.nan_to_num(data, nan=-9999.0), 1)

    raw_path.unlink()
    print(f"  DTM: {output_path.name} ({px_w}x{px_h}, {width:.0f}x{height:.0f}m)")
    return px_w, px_h


def download_ahn4_dsm(
    bbox: tuple, output_path: Path, size_px: int = 2048
) -> tuple[int, int]:
    """Download AHN4 DSM (oppervlaktemodel) via ArcGIS ImageServer.

    Het DSM bevat de hoogte inclusief objecten (bomen, gebouwen).
    DSM - DTM = nDSM (genormaliseerd hoogte model) dat vegetatie en objecten toont.

    Parameters en returns identiek aan download_ahn4_dtm.
    """
    minx, miny, maxx, maxy = bbox
    width = maxx - minx
    height = maxy - miny
    aspect = width / height

    if aspect >= 1:
        px_w = size_px
        px_h = max(64, int(size_px / aspect))
    else:
        px_h = size_px
        px_w = max(64, int(size_px * aspect))

    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "bboxSR": "28992",
        "imageSR": "28992",
        "size": f"{px_w},{px_h}",
        "format": "tiff",
        "pixelType": "F32",
        "noData": "-9999",
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }

    resp = requests.get(f"{AHN4_DSM_URL}/exportImage", params=params, timeout=120)
    resp.raise_for_status()

    if resp.headers.get("content-type", "").startswith("application/json"):
        raise RuntimeError(f"AHN4 DSM server error: {resp.json()}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_suffix(".raw.tif")
    raw_path.write_bytes(resp.content)

    transform = from_bounds(minx, miny, maxx, maxy, px_w, px_h)
    with rasterio.open(raw_path) as src:
        data = src.read(1).astype(np.float32)

    data[data < -100] = np.nan

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=px_h,
        width=px_w,
        count=1,
        dtype="float32",
        crs="EPSG:28992",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(np.nan_to_num(data, nan=-9999.0), 1)

    raw_path.unlink()
    print(f"  DSM: {output_path.name} ({px_w}x{px_h})")
    return px_w, px_h


def download_luchtfoto(
    bbox: tuple, output_path: Path, px_w: int, px_h: int
) -> None:
    """Download luchtfoto (RGB) via ArcGIS MapServer.

    Parameters
    ----------
    bbox : tuple
        (minx, miny, maxx, maxy) in EPSG:28992.
    output_path : Path
        Uitvoerpad voor het GeoTIFF.
    px_w, px_h : int
        Gewenste afmetingen in pixels.
    """
    minx, miny, maxx, maxy = bbox

    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "bboxSR": "28992",
        "imageSR": "28992",
        "size": f"{px_w},{px_h}",
        "format": "png",
        "f": "image",
    }

    resp = requests.get(LUCHTFOTO_URL, params=params, timeout=120)
    resp.raise_for_status()

    if "image" not in resp.headers.get("content-type", ""):
        raise RuntimeError(f"Luchtfoto error: {resp.text[:300]}")

    img = Image.open(BytesIO(resp.content)).convert("RGB")
    rgb = np.array(img.resize((px_w, px_h), Image.BILINEAR))

    transform = from_bounds(minx, miny, maxx, maxy, px_w, px_h)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=px_h,
        width=px_w,
        count=3,
        dtype="uint8",
        crs="EPSG:28992",
        transform=transform,
    ) as dst:
        for band in range(3):
            dst.write(rgb[:, :, band], band + 1)

    print(f"  Luchtfoto: {output_path.name}")


def download_bgt_waterdeel(
    bbox: tuple, limit: int = 1000
) -> list[dict]:
    """Download BGT waterdeel polygonen via OGC Features API.

    Parameters
    ----------
    bbox : tuple
        (minx, miny, maxx, maxy) in EPSG:28992.
    limit : int
        Maximum aantal features per request.

    Returns
    -------
    list[dict]
        Lijst met dicts: {'geometry': Shapely Polygon in EPSG:28992, 'type': str}
    """
    from pyproj import Transformer
    from shapely.ops import transform as shp_transform

    # BGT OGC API verwacht WGS84 bbox en geeft WGS84 terug
    to_wgs = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
    to_rd = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)

    wgs_minx, wgs_miny = to_wgs.transform(bbox[0], bbox[1])
    wgs_maxx, wgs_maxy = to_wgs.transform(bbox[2], bbox[3])

    url = f"{BGT_OGC_URL}/collections/waterdeel/items"
    all_features = []
    next_url = None

    params = {
        "bbox": f"{wgs_minx},{wgs_miny},{wgs_maxx},{wgs_maxy}",
        "limit": limit,
        "f": "json",
    }

    while True:
        if next_url:
            resp = requests.get(next_url, timeout=60)
        else:
            resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        features = data.get("features", [])

        for feat in features:
            geom_wgs = shape(feat["geometry"])
            geom_rd = shp_transform(to_rd.transform, geom_wgs)
            props = feat.get("properties", {})
            water_type = props.get("plus_type", props.get("bgt_type", "water"))
            all_features.append({"geometry": geom_rd, "type": water_type})

        # Cursor-based paging via 'next' link
        next_url = None
        for link in data.get("links", []):
            if link.get("rel") == "next":
                next_url = link["href"]
                break

        if not next_url or len(features) == 0:
            break

    print(f"  BGT waterdeel: {len(all_features)} polygonen")
    return all_features


def download_bgt_ondersteunend_waterdeel(
    bbox: tuple, limit: int = 1000
) -> list[dict]:
    """Download BGT ondersteunend waterdeel (oevers, slootkanten).

    Parameters
    ----------
    bbox : tuple
        (minx, miny, maxx, maxy) in EPSG:28992.

    Returns
    -------
    list[dict]
        Lijst met dicts: {'geometry': Shapely Polygon in EPSG:28992, 'type': str}
    """
    from pyproj import Transformer
    from shapely.ops import transform as shp_transform

    to_wgs = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
    to_rd = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)

    wgs_minx, wgs_miny = to_wgs.transform(bbox[0], bbox[1])
    wgs_maxx, wgs_maxy = to_wgs.transform(bbox[2], bbox[3])

    url = f"{BGT_OGC_URL}/collections/ondersteunendwaterdeel/items"
    all_features = []
    next_url = None

    params = {
        "bbox": f"{wgs_minx},{wgs_miny},{wgs_maxx},{wgs_maxy}",
        "limit": limit,
        "f": "json",
    }

    while True:
        if next_url:
            resp = requests.get(next_url, timeout=60)
        else:
            resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        features = data.get("features", [])

        for feat in features:
            geom_wgs = shape(feat["geometry"])
            geom_rd = shp_transform(to_rd.transform, geom_wgs)
            props = feat.get("properties", {})
            water_type = props.get("plus_type", props.get("bgt_type", "oever"))
            all_features.append({"geometry": geom_rd, "type": water_type})

        next_url = None
        for link in data.get("links", []):
            if link.get("rel") == "next":
                next_url = link["href"]
                break

        if not next_url or len(features) == 0:
            break

    if all_features:
        print(f"  BGT ondersteunend waterdeel: {len(all_features)} polygonen")
    return all_features


def load_dtb_lines(
    geojson_path: str | Path,
    bbox: tuple | None = None,
) -> dict[str, list]:
    """Laad DTB referentielijnen uit een GeoJSON export.

    Classificeert talud(bovenkant) en talud(onderkant) lijnen als
    binnen- of buitenzijde op basis van positie t.o.v. de kruinlijn.

    Parameters
    ----------
    geojson_path : str | Path
        Pad naar het DTB GeoJSON bestand (EPSG:28992).
    bbox : tuple | None
        Optioneel (minx, miny, maxx, maxy) om te filteren.

    Returns
    -------
    dict[str, list[LineString]]
        Keys: 'kruinlijn', 'talud_boven', 'talud_onder'.
        Waarden: lijsten van LineStrings in EPSG:28992.
    """
    import json
    from shapely.geometry import LineString, shape as shp_shape, box as shp_box

    path = Path(geojson_path)
    with open(path) as f:
        data = json.load(f)

    clip_box = shp_box(*bbox) if bbox else None

    result = {"kruinlijn": [], "talud_boven": [], "talud_onder": []}

    for feat in data.get("features", []):
        omschr = feat.get("properties", {}).get("omschr", "")
        geom = shp_shape(feat["geometry"])

        if geom.is_empty:
            continue

        # Multi → single
        if geom.geom_type == "MultiLineString":
            parts = list(geom.geoms)
        elif geom.geom_type == "LineString":
            parts = [geom]
        else:
            continue

        for line in parts:
            if line.length < 1:
                continue

            if clip_box is not None:
                line = line.intersection(clip_box)
                if line.is_empty:
                    continue
                if line.geom_type == "MultiLineString":
                    line = max(line.geoms, key=lambda g: g.length)
                elif line.geom_type != "LineString":
                    continue

            if omschr == "kruinlijn":
                result["kruinlijn"].append(line)
            elif omschr == "talud(bovenkant)":
                result["talud_boven"].append(line)
            elif omschr == "talud(onderkant)":
                result["talud_onder"].append(line)

    for key, lines in result.items():
        total_len = sum(l.length for l in lines)
        if lines:
            print(f"  DTB {key}: {len(lines)} lijnen, {total_len:.0f}m")

    return result


def classify_dtb_sides(
    dtb_lines: dict[str, list],
    dijk_richting: str = "auto",
) -> dict[str, list]:
    """Classificeer DTB talud-lijnen in binnen/buiten-zijde.

    De DTB heeft talud(bovenkant) en talud(onderkant) maar geeft niet
    aan welke de binnenzijde (polderzijde) en buitenzijde (waterzijde) is.

    Strategie: per talud(onderkant) lijn, bepaal of deze ten noorden of
    ten zuiden van de dichtstbijzijnde kruinlijn ligt. De zijde met de
    laagste gemiddelde Z-hoogte is de buitenzijde (waterzijde).

    Parameters
    ----------
    dtb_lines : dict
        Output van load_dtb_lines().
    dijk_richting : str
        'auto': detecteer automatisch, 'noord': water aan noordkant,
        'zuid': water aan zuidkant.

    Returns
    -------
    dict[str, list[LineString]]
        Keys: 'binnenkruin', 'buitenkruin', 'binnenteen', 'buitenteen'.
        Waarden: lijsten van LineStrings.
    """
    from shapely.ops import unary_union

    kruin_lines = dtb_lines.get("kruinlijn", [])
    boven_lines = dtb_lines.get("talud_boven", [])
    onder_lines = dtb_lines.get("talud_onder", [])

    # kruinlijn + talud(bovenkant) zijn allebei kruin-achtig
    all_kruin = kruin_lines + boven_lines
    all_teen = onder_lines

    if not all_kruin and not all_teen:
        return {"binnenkruin": [], "buitenkruin": [],
                "binnenteen": [], "buitenteen": []}

    # Bepaal gemiddelde kruin-Y per lokale groep
    # Gebruik unary_union centroid als globale referentie
    if all_kruin:
        kruin_union = unary_union(all_kruin)
        kruin_y = kruin_union.centroid.y
    else:
        kruin_y = np.mean([np.mean([c[1] for c in l.coords]) for l in all_teen])

    # Splits teenlijnen in noord/zuid t.o.v. lokale kruin
    north_teens = []
    south_teens = []
    for line in all_teen:
        line_y = np.mean([c[1] for c in line.coords])
        if line_y > kruin_y:
            north_teens.append(line)
        else:
            south_teens.append(line)

    # Bepaal welke kant het water is
    if dijk_richting == "auto" and north_teens and south_teens:
        def _avg_z(lines):
            zs = [c[2] for l in lines for c in l.coords if len(c) >= 3]
            return np.mean(zs) if zs else 0
        north_z = _avg_z(north_teens)
        south_z = _avg_z(south_teens)

        if north_z < south_z:
            buiten_teens = north_teens
            binnen_teens = south_teens
            print(f"  DTB zijde-detectie: water aan noordkant (N:{north_z:.1f}m < Z:{south_z:.1f}m)")
        else:
            buiten_teens = south_teens
            binnen_teens = north_teens
            print(f"  DTB zijde-detectie: water aan zuidkant (Z:{south_z:.1f}m < N:{north_z:.1f}m)")
    elif dijk_richting == "noord":
        buiten_teens = north_teens
        binnen_teens = south_teens
    elif dijk_richting == "zuid":
        buiten_teens = south_teens
        binnen_teens = north_teens
    else:
        buiten_teens = all_teen
        binnen_teens = []

    # Kruinlijnen: gebruik alles als zowel binnen- als buitenkruin
    # (DTB maakt geen onderscheid, de kruin is de kruin)
    total_kruin = sum(l.length for l in all_kruin)
    total_binnen = sum(l.length for l in binnen_teens)
    total_buiten = sum(l.length for l in buiten_teens)
    print(f"  DTB totaal: kruin {total_kruin:.0f}m, binnenteen {total_binnen:.0f}m, buitenteen {total_buiten:.0f}m")

    return {
        "binnenkruin": all_kruin,
        "buitenkruin": all_kruin,
        "binnenteen": binnen_teens,
        "buitenteen": buiten_teens,
    }


def _ahn4_subtiles_for_bbox(bbox: tuple) -> list[tuple[str, str]]:
    """Bepaal welke AHN4 LAZ subtiles een bbox dekken.

    Returns lijst van (tile_name, download_url) tuples.
    """
    import json
    import urllib.request

    # Haal kaartbladindex op van PDOK
    idx_url = "https://service.pdok.nl/rws/ahn/atom/downloads/dtm_05m/kaartbladindex.json"
    req = urllib.request.Request(idx_url, headers={"User-Agent": "geoTooling/1.0"})
    resp = urllib.request.urlopen(req, timeout=30)
    data = json.loads(resp.read())

    from shapely.geometry import shape as shp_shape

    target = box(*bbox)
    results = []

    for feat in data["features"]:
        geom = shp_shape(feat["geometry"])
        if not geom.intersects(target):
            continue

        blad_nr = feat["properties"]["kaartbladNr"]  # e.g. "M_39GN1"
        # Haal blad-code uit: M_39GN1 -> 39GN1
        blad = blad_nr.replace("M_", "")
        blad_bounds = geom.bounds

        # Bereken welke subtiles (1x1.25km) nodig zijn
        bx_min, by_min, bx_max, by_max = blad_bounds
        dx = (bx_max - bx_min) / 5   # 1000m
        dy = (by_max - by_min) / 5    # 1250m

        for row in range(5):
            for col in range(5):
                st_x_min = bx_min + col * dx
                st_x_max = st_x_min + dx
                st_y_max = by_max - row * dy
                st_y_min = st_y_max - dy

                st_box = box(st_x_min, st_y_min, st_x_max, st_y_max)
                # Subtiles hebben 25m overlap, dus iets ruimer testen
                if st_box.buffer(30).intersects(target):
                    subtile_nr = row * 5 + col + 1
                    name = f"{blad}_{subtile_nr:02d}"
                    url = f"https://geotiles.citg.tudelft.nl/AHN4_T/{name}.LAZ"
                    results.append((name, url))

    return results


def download_ahn4_pointcloud_dtm(
    bbox: tuple,
    output_path: Path,
    resolution: float = 0.5,
    cache_dir: Path | None = None,
    classes: list[int] | None = None,
) -> tuple[int, int]:
    """Download AHN4 puntenwolk en maak DTM uit grondpunten.

    Parameters
    ----------
    bbox : tuple
        (minx, miny, maxx, maxy) in EPSG:28992.
    output_path : Path
        Uitvoerpad voor het GeoTIFF.
    resolution : float
        Pixelgrootte in meters (default 0.5m).
    cache_dir : Path | None
        Map om LAZ bestanden te cachen. None = data/pointcloud.
    classes : list[int] | None
        LAS classificaties. None = [2] (alleen grond).
        AHN4: 2=ground, 6=building, 1=unclassified

    Returns
    -------
    tuple[int, int]
        (breedte_px, hoogte_px)
    """
    import urllib.request

    import laspy
    from scipy.interpolate import griddata

    if classes is None:
        classes = [2]  # Alleen grondpunten

    if cache_dir is None:
        cache_dir = Path("data/pointcloud")
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    minx, miny, maxx, maxy = bbox

    # Bepaal welke subtiles nodig zijn
    subtiles = _ahn4_subtiles_for_bbox(bbox)
    if not subtiles:
        raise RuntimeError(f"Geen AHN4 puntenwolk-tiles gevonden voor bbox {bbox}")

    print(f"  Puntenwolk: {len(subtiles)} subtile(s) nodig")

    # Download en verzamel grondpunten
    all_x, all_y, all_z = [], [], []
    for name, url in subtiles:
        laz_path = cache_dir / f"{name}.LAZ"
        if not laz_path.exists():
            print(f"    Downloaden {name}.LAZ ...")
            req = urllib.request.Request(url, headers={"User-Agent": "geoTooling/1.0"})
            resp = urllib.request.urlopen(req, timeout=600)
            laz_path.write_bytes(resp.read())
            size_mb = laz_path.stat().st_size / 1024 / 1024
            print(f"    {name}.LAZ: {size_mb:.0f} MB")
        else:
            size_mb = laz_path.stat().st_size / 1024 / 1024
            print(f"    {name}.LAZ: cached ({size_mb:.0f} MB)")

        # Lees en filter
        las = laspy.read(str(laz_path))
        mask = np.isin(las.classification, classes)
        x = np.array(las.x[mask])
        y = np.array(las.y[mask])
        z = np.array(las.z[mask])

        # Clip op bbox
        clip = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)
        all_x.append(x[clip])
        all_y.append(y[clip])
        all_z.append(z[clip])

    x = np.concatenate(all_x)
    y = np.concatenate(all_y)
    z = np.concatenate(all_z)
    print(f"  {len(x):,} grondpunten in bbox")

    if len(x) < 100:
        raise RuntimeError(f"Te weinig punten ({len(x)}) in bbox")

    # Interpoleer naar grid
    cols = int(np.ceil((maxx - minx) / resolution))
    rows = int(np.ceil((maxy - miny) / resolution))

    xi = np.linspace(minx + resolution / 2, maxx - resolution / 2, cols)
    yi = np.linspace(maxy - resolution / 2, miny + resolution / 2, rows)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    grid = griddata((x, y), z, (xi_grid, yi_grid), method="linear", fill_value=np.nan)

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
    transform = from_bounds(minx, miny, maxx, maxy, cols, rows)
    with rasterio.open(
        output_path, "w", driver="GTiff",
        height=rows, width=cols, count=1, dtype="float32",
        crs="EPSG:28992", transform=transform, compress="deflate",
    ) as dst:
        dst.write(grid, 1)

    print(f"  PC-DTM: {output_path.name} ({cols}x{rows}, {resolution}m)")
    return cols, rows


def download_ahn4_pointcloud_dsm(
    bbox: tuple,
    output_path: Path,
    resolution: float = 0.5,
    cache_dir: Path | None = None,
) -> tuple[int, int]:
    """Download AHN4 puntenwolk en maak DSM (alle punten behalve water).

    Identiek aan download_ahn4_pointcloud_dtm maar met klassen 1+2+6
    (unclassified + ground + building).
    """
    return download_ahn4_pointcloud_dtm(
        bbox, output_path, resolution, cache_dir,
        classes=[1, 2, 6],  # Alles behalve water (9)
    )


def load_hdsr_kniklijnen(
    geojson_path: str | Path,
    bbox: tuple | None = None,
) -> dict[str, list]:
    """Laad HDSR kniklijnen uit GeoJSON (EPSG:4326 → EPSG:28992).

    HDSR kniklijnen hebben een 'type' veld dat direct de profiellijn aangeeft:
    15=teen_binnen, 16=teen_buiten, 17=kruin_binnen, 18=kruin_buiten,
    60=berm_binnen, 61=berm_buiten, 62=insteek_buiten, 63=insteek_binnen.

    Parameters
    ----------
    geojson_path : str | Path
        Pad naar het HDSR GeoJSON (EPSG:4326).
    bbox : tuple | None
        Optioneel (minx, miny, maxx, maxy) in EPSG:28992 om te filteren.

    Returns
    -------
    dict[str, list[LineString]]
        Keys: 'binnenkruin', 'buitenkruin', 'binnenteen', 'buitenteen',
        'binnenberm', 'buitenberm', 'insteek'.
    """
    import json
    from pyproj import Transformer
    from shapely.geometry import LineString, shape as shp_shape, box as shp_box

    path = Path(geojson_path)
    with open(path) as f:
        data = json.load(f)

    to_rd = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
    clip_box = shp_box(*bbox) if bbox else None

    # HDSR type codes → onze referentielijn-namen
    type_map = {
        15: "binnenteen", 16: "buitenteen",
        17: "binnenkruin", 18: "buitenkruin",
        60: "binnenberm", 61: "buitenberm",
        62: "insteek", 63: "insteek",
    }

    result = {
        "binnenkruin": [], "buitenkruin": [],
        "binnenteen": [], "buitenteen": [],
        "binnenberm": [], "buitenberm": [],
        "insteek": [],
    }

    for feat in data.get("features", []):
        props = feat.get("properties", {})
        line_type = props.get("type")
        target_key = type_map.get(line_type)
        if target_key is None:
            continue

        geom = shp_shape(feat["geometry"])
        if geom.is_empty:
            continue

        # Converteer naar lijsten van LineStrings
        if geom.geom_type == "MultiLineString":
            parts = list(geom.geoms)
        elif geom.geom_type == "LineString":
            parts = [geom]
        else:
            continue

        for line_wgs in parts:
            if line_wgs.length < 0.00001:  # WGS84 threshold
                continue

            # Transform naar RD
            coords_rd = [to_rd.transform(x, y) for x, y in line_wgs.coords]
            line = LineString(coords_rd)

            if line.length < 1:
                continue

            if clip_box is not None:
                line = line.intersection(clip_box)
                if line.is_empty:
                    continue
                if line.geom_type == "MultiLineString":
                    line = max(line.geoms, key=lambda g: g.length)
                elif line.geom_type != "LineString":
                    continue

            result[target_key].append(line)

    for key, lines in result.items():
        if lines:
            total_len = sum(l.length for l in lines)
            print(f"  HDSR {key}: {len(lines)} lijnen, {total_len:.0f}m")

    return result
