"""Hybride pipeline: morfologische analyse + deep learning voor kruinlijndetectie."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import rowcol
from shapely.geometry import LineString, mapping

from .morpho import generate_cross_profiles, extract_profile_elevations, detect_crest_points, crest_points_to_line


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


def generate_training_labels(
    dtm_path: str | Path,
    centerline: LineString,
    output_labels_path: str | Path,
    crest_buffer: float = 1.5,
    talud_width: float = 10.0,
    teen_buffer: float = 2.0,
) -> np.ndarray:
    """Genereer segmentatie-labels op basis van morfologische kruinlijndetectie.

    Dit is de brug tussen de klassieke en DL-benadering: gebruik de morfologische
    methode om automatisch trainingsdata te genereren voor het segmentatiemodel.

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
        Geschatte breedte (m) van het talud aan elke zijde.
    teen_buffer : float
        Breedte (m) van de teenzone.

    Returns
    -------
    np.ndarray
        Label-array (0=achtergrond, 1=kruin, 2=talud_in, 3=teen_in, 4=talud_uit, 5=teen_uit).
    """
    # Stap 1: detecteer kruinlijn via morfologische methode
    crest_line, _ = morpho_pipeline(dtm_path, centerline, spacing=2.0, width=50.0)

    if crest_line is None:
        raise RuntimeError("Kon geen kruinlijn detecteren — kan geen labels genereren.")

    with rasterio.open(dtm_path) as src:
        shape = (src.height, src.width)
        transform = src.transform
        profile = src.profile.copy()

    # Stap 2: maak zones via buffering
    # De kruin is een smalle zone rond de kruinlijn
    crest_zone = crest_line.buffer(crest_buffer)

    # Talud = zone tussen kruin en teen, we benaderen dit door de kruin
    # verder te bufferen en het verschil te nemen.
    inner_zone = crest_line.buffer(crest_buffer + talud_width)
    talud_zone = inner_zone.difference(crest_zone)

    outer_zone = inner_zone.buffer(teen_buffer)
    teen_zone = outer_zone.difference(inner_zone)

    # Stap 3: rasterize — van buiten naar binnen (later overschrijft eerder)
    # We gebruiken een vereenvoudigd model: links = binnen, rechts = buiten
    # In werkelijkheid zou je de zijde moeten bepalen t.o.v. de dijk-orientatie
    shapes_and_labels = [
        (mapping(teen_zone), 3),       # teen (voorlopig als 'binnen')
        (mapping(talud_zone), 2),      # talud (voorlopig als 'binnen')
        (mapping(crest_zone), 1),      # kruin
    ]

    labels = rasterize(
        shapes_and_labels,
        out_shape=shape,
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
