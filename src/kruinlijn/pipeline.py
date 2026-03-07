"""Hybride pipeline: morfologische analyse + deep learning voor kruinlijndetectie."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import rowcol
from shapely.geometry import LineString, mapping

from .morpho import (
    generate_cross_profiles,
    extract_profile_elevations,
    detect_crest_points,
    crest_points_to_line,
    detect_knikpunten,
    knikpunten_to_lines,
    KNIKPUNT_TYPES,
)


def morpho_pipeline(
    dtm_path: str | Path,
    centerline: LineString,
    output_gpkg: str | Path | None = None,
    spacing: float = 5.0,
    width: float = 40.0,
    smooth_sigma: float = 2.0,
    method: str = "curvature",
) -> tuple[LineString | None, gpd.GeoDataFrame]:
    """Volledige morfologische kruinlijndetectie.

    Parameters
    ----------
    dtm_path : str | Path
        Pad naar het DTM GeoTIFF.
    centerline : LineString
        Ruwe hartlijn van de dijk.
    output_gpkg : str | Path | None
        Optioneel pad om resultaten als GeoPackage op te slaan.
    spacing, width, smooth_sigma, method
        Parameters voor profielgeneratie en kruindetectie.

    Returns
    -------
    crest_line : LineString | None
        De gedetecteerde kruinlijn.
    points_gdf : GeoDataFrame
        GeoDataFrame met alle gedetecteerde kruinpunten.
    """
    # 1. Genereer dwarsprofielen
    profiles = generate_cross_profiles(centerline, spacing=spacing, width=width)

    # 2. Sample hoogtewaardes
    profiles = extract_profile_elevations(profiles, str(dtm_path))

    # 3. Detecteer kruinpunten
    profiles = detect_crest_points(profiles, smooth_sigma=smooth_sigma, method=method)

    # 4. Verbind tot kruinlijn
    crest_line = crest_points_to_line(profiles)

    # 5. Maak GeoDataFrame van kruinpunten
    crest_data = []
    for p in profiles:
        if p.get("crest_xy") is not None:
            crest_data.append(
                {
                    "distance_along": p["distance"],
                    "z": p["crest_z"],
                    "geometry": gpd.points_from_xy([p["crest_xy"][0]], [p["crest_xy"][1]])[0],
                }
            )

    points_gdf = gpd.GeoDataFrame(crest_data)

    # CRS overnemen van DTM
    with rasterio.open(dtm_path) as src:
        if src.crs:
            points_gdf = points_gdf.set_crs(src.crs)

    if output_gpkg:
        output_gpkg = Path(output_gpkg)
        points_gdf.to_file(output_gpkg, layer="kruinpunten", driver="GPKG")
        if crest_line is not None:
            line_gdf = gpd.GeoDataFrame(
                [{"geometry": crest_line}], crs=points_gdf.crs
            )
            line_gdf.to_file(output_gpkg, layer="kruinlijn", driver="GPKG")
        print(f"Resultaten opgeslagen: {output_gpkg}")

    return crest_line, points_gdf


def kniklijnen_pipeline(
    dtm_path: str | Path,
    centerline: LineString,
    output_gpkg: str | Path | None = None,
    spacing: float = 5.0,
    width: float = 40.0,
    smooth_sigma: float = 2.0,
    method: str = "curvature",
    water_side: str | None = None,
) -> tuple[dict[str, LineString | None], gpd.GeoDataFrame]:
    """Volledige kniklijnendetectie: kruin + binnenteen, buitenteen, kruinranden.

    Parameters
    ----------
    dtm_path : str | Path
        Pad naar het DTM GeoTIFF.
    centerline : LineString
        Ruwe hartlijn van de dijk.
    output_gpkg : str | Path | None
        Optioneel pad om resultaten als GeoPackage op te slaan.
    spacing, width, smooth_sigma, method
        Parameters voor profielgeneratie en kruindetectie.

    Returns
    -------
    kniklijnen : dict[str, LineString | None]
        Dict met lijnen per type: 'kruin', 'binnenteen', 'buitenteen',
        'binnenkruin', 'buitenkruin'.
    points_gdf : GeoDataFrame
        GeoDataFrame met alle gedetecteerde knikpunten.
    """
    # 1. Genereer dwarsprofielen
    profiles = generate_cross_profiles(centerline, spacing=spacing, width=width)

    # 2. Sample hoogtewaardes
    profiles = extract_profile_elevations(profiles, str(dtm_path))

    # 3. Detecteer kruinpunten
    profiles = detect_crest_points(profiles, smooth_sigma=smooth_sigma, method=method)

    # 4. Detecteer knikpunten
    profiles = detect_knikpunten(profiles, smooth_sigma=smooth_sigma + 1.0, water_side=water_side)

    # 5. Maak lijnen
    crest_line = crest_points_to_line(profiles)
    knik_lines = knikpunten_to_lines(profiles)
    kniklijnen = {"kruin": crest_line, **knik_lines}

    # 6. Maak GeoDataFrame van alle knikpunten
    all_types = ["crest"] + KNIKPUNT_TYPES
    type_labels = ["kruin"] + KNIKPUNT_TYPES
    xy_keys = ["crest_xy"] + [f"{kt}_xy" for kt in KNIKPUNT_TYPES]
    z_keys = ["crest_z"] + [f"{kt}_z" for kt in KNIKPUNT_TYPES]

    knik_data = []
    for p in profiles:
        for label, xy_key, z_key in zip(type_labels, xy_keys, z_keys):
            xy = p.get(xy_key)
            z = p.get(z_key)
            if xy is not None and z is not None:
                knik_data.append(
                    {
                        "type": label,
                        "distance_along": p["distance"],
                        "z": z,
                        "geometry": gpd.points_from_xy([xy[0]], [xy[1]])[0],
                    }
                )

    points_gdf = gpd.GeoDataFrame(knik_data)

    # CRS overnemen van DTM
    with rasterio.open(dtm_path) as src:
        if src.crs:
            points_gdf = points_gdf.set_crs(src.crs)

    if output_gpkg:
        output_gpkg = Path(output_gpkg)
        points_gdf.to_file(output_gpkg, layer="knikpunten", driver="GPKG")
        for naam, lijn in kniklijnen.items():
            if lijn is not None:
                line_gdf = gpd.GeoDataFrame(
                    [{"type": naam, "geometry": lijn}], crs=points_gdf.crs
                )
                line_gdf.to_file(output_gpkg, layer=f"kniklijn_{naam}", driver="GPKG")
        print(f"Kniklijnen opgeslagen: {output_gpkg}")

    return kniklijnen, points_gdf


def generate_training_labels(
    dtm_path: str | Path,
    centerline: LineString,
    output_labels_path: str | Path,
    crest_buffer: float = 1.5,
    talud_width: float = 12.0,
    teen_buffer: float = 2.0,
    water_side: str | None = None,
) -> np.ndarray:
    """Genereer segmentatie-labels op basis van morfologische kniklijnendetectie.

    Gebruikt de volledige kniklijnen-pipeline om binnen- en buitenzijde
    correct te labelen. Zones worden bepaald door de gedetecteerde kniklijnen
    (binnenkruin, buitenkruin, binnenteen, buitenteen).

    Parameters
    ----------
    dtm_path : str | Path
        Pad naar het DTM GeoTIFF.
    centerline : LineString
        Ruwe hartlijn van de dijk.
    output_labels_path : str | Path
        Pad voor het output label-raster.
    crest_buffer : float
        Breedte (m) van de kruinzone rond de gedetecteerde kruinlijn.
    talud_width : float
        Fallback breedte (m) als een kniklijn niet gedetecteerd is.
    teen_buffer : float
        Breedte (m) van de teenzone rond de teenlijnen.
    water_side : str | None
        'left' of 'right' om buitenzijde te forceren.

    Returns
    -------
    np.ndarray
        Label-array (0=achtergrond, 1=kruin, 2=talud_binnen, 3=teen_binnen,
        4=talud_buiten, 5=teen_buiten).
    """
    from shapely.ops import split
    from shapely.geometry import Polygon, MultiPolygon

    # Stap 1: detecteer alle kniklijnen
    kniklijnen, _ = kniklijnen_pipeline(
        dtm_path, centerline, spacing=2.0, width=60.0,
        smooth_sigma=2.0, water_side=water_side,
    )

    crest_line = kniklijnen.get("kruin")
    if crest_line is None:
        raise RuntimeError("Kon geen kruinlijn detecteren — kan geen labels genereren.")

    with rasterio.open(dtm_path) as src:
        raster_shape = (src.height, src.width)
        transform = src.transform
        profile = src.profile.copy()

    # Stap 2: bouw zones uit kniklijnen
    binnenkruin = kniklijnen.get("binnenkruin")
    buitenkruin = kniklijnen.get("buitenkruin")
    binnenteen = kniklijnen.get("binnenteen")
    buitenteen = kniklijnen.get("buitenteen")

    # Kruin zone: gebied tussen binnenkruin en buitenkruin
    if binnenkruin and buitenkruin:
        kruin_zone = crest_line.buffer(crest_buffer)
        bk_buf = binnenkruin.buffer(0.5)
        buk_buf = buitenkruin.buffer(0.5)
        kruin_zone = kruin_zone.union(bk_buf).union(buk_buf).convex_hull.intersection(
            crest_line.buffer(crest_buffer + 3.0)
        )
    else:
        kruin_zone = crest_line.buffer(crest_buffer)

    # Talud binnen: zone tussen binnenkruin en binnenteen
    if binnenkruin and binnenteen:
        talud_binnen_zone = binnenkruin.buffer(talud_width).intersection(
            binnenteen.buffer(talud_width)
        )
        talud_binnen_zone = talud_binnen_zone.difference(kruin_zone)
    elif binnenkruin:
        buf = binnenkruin.buffer(talud_width)
        talud_binnen_zone = buf.difference(kruin_zone)
    else:
        buf = crest_line.buffer(crest_buffer + talud_width)
        talud_binnen_zone = buf.difference(kruin_zone)
        # Neem alleen de binnenzijde (beperkt tot halve buffer)
        half = crest_line.buffer(crest_buffer + talud_width / 2)
        talud_binnen_zone = talud_binnen_zone.intersection(half)

    # Talud buiten: zone tussen buitenkruin en buitenteen
    if buitenkruin and buitenteen:
        talud_buiten_zone = buitenkruin.buffer(talud_width).intersection(
            buitenteen.buffer(talud_width)
        )
        talud_buiten_zone = talud_buiten_zone.difference(kruin_zone)
    elif buitenkruin:
        buf = buitenkruin.buffer(talud_width)
        talud_buiten_zone = buf.difference(kruin_zone)
    else:
        buf = crest_line.buffer(crest_buffer + talud_width)
        talud_buiten_zone = buf.difference(kruin_zone)

    # Verwijder overlap tussen talud_binnen en talud_buiten
    talud_binnen_zone = talud_binnen_zone.difference(talud_buiten_zone.intersection(talud_binnen_zone).buffer(-0.1))

    # Teen zones: smalle zone rond teenlijnen
    teen_binnen_zone = binnenteen.buffer(teen_buffer) if binnenteen else None
    teen_buiten_zone = buitenteen.buffer(teen_buffer) if buitenteen else None

    # Verwijder teen overlap met talud
    if teen_binnen_zone:
        teen_binnen_zone = teen_binnen_zone.difference(kruin_zone)
    if teen_buiten_zone:
        teen_buiten_zone = teen_buiten_zone.difference(kruin_zone)

    # Stap 3: rasterize — van buiten naar binnen (later overschrijft eerder)
    shapes_and_labels = []
    if teen_buiten_zone and not teen_buiten_zone.is_empty:
        shapes_and_labels.append((mapping(teen_buiten_zone), 5))
    if teen_binnen_zone and not teen_binnen_zone.is_empty:
        shapes_and_labels.append((mapping(teen_binnen_zone), 3))
    if not talud_buiten_zone.is_empty:
        shapes_and_labels.append((mapping(talud_buiten_zone), 4))
    if not talud_binnen_zone.is_empty:
        shapes_and_labels.append((mapping(talud_binnen_zone), 2))
    shapes_and_labels.append((mapping(kruin_zone), 1))

    labels = rasterize(
        shapes_and_labels,
        out_shape=raster_shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    # Stap 4: schrijf label-raster
    profile.update(dtype="uint8", count=1, nodata=0)
    with rasterio.open(output_labels_path, "w", **profile) as dst:
        dst.write(labels, 1)

    print(f"Labels gegenereerd: {output_labels_path}")
    return labels
