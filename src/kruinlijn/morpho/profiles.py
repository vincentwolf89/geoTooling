"""Genereren en samplen van dwarsprofielen langs een dijklijn."""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import rowcol
from shapely.geometry import LineString, Point


def generate_cross_profiles(
    centerline: LineString,
    spacing: float = 5.0,
    width: float = 40.0,
    sample_dist: float = 0.5,
) -> list[dict]:
    """Genereer dwarsprofielen loodrecht op een dijkhartlijn.

    Parameters
    ----------
    centerline : LineString
        De (ruwe) dijkhartlijn waarlangs profielen worden gegenereerd.
    spacing : float
        Afstand (m) tussen opeenvolgende profielen langs de lijn.
    width : float
        Totale breedte (m) van elk dwarsprofiel (symmetrisch rond de hartlijn).
    sample_dist : float
        Afstand (m) tussen samplepoints op het dwarsprofiel.

    Returns
    -------
    list[dict]
        Elke dict bevat:
        - 'distance': afstand langs de hartlijn
        - 'center': (x, y) middelpunt
        - 'points': np.ndarray met shape (n, 2) — xy-coordinaten van het profiel
        - 'line': LineString van het dwarsprofiel
    """
    total_length = centerline.length
    distances = np.arange(0, total_length, spacing)

    profiles = []
    for d in distances:
        pt = centerline.interpolate(d)
        # Bepaal loodrechte richting via tangent
        tangent = _tangent_at(centerline, d)
        normal = np.array([-tangent[1], tangent[0]])

        half = width / 2.0
        offsets = np.arange(-half, half + sample_dist, sample_dist)
        coords = np.array([pt.x, pt.y]) + np.outer(offsets, normal)

        profiles.append(
            {
                "distance": d,
                "center": (pt.x, pt.y),
                "points": coords,
                "offsets": offsets,
                "line": LineString(coords),
            }
        )
    return profiles


def extract_profile_elevations(
    profiles: list[dict],
    dtm_path: str,
    band: int = 1,
    nodata_val: float = np.nan,
) -> list[dict]:
    """Sample hoogtewaardes uit een DTM raster voor elk dwarsprofiel.

    Parameters
    ----------
    profiles : list[dict]
        Profielen zoals gegenereerd door ``generate_cross_profiles``.
    dtm_path : str
        Pad naar het DTM GeoTIFF-bestand.
    band : int
        Rasterband om te lezen (default 1).
    nodata_val : float
        Waarde om toe te kennen als een punt buiten het raster valt.

    Returns
    -------
    list[dict]
        Dezelfde profielen, aangevuld met:
        - 'elevations': np.ndarray met hoogtewaardes per samplepoint
    """
    with rasterio.open(dtm_path) as src:
        data = src.read(band)
        transform = src.transform
        nodata = src.nodata

        for profile in profiles:
            pts = profile["points"]
            elevations = np.full(len(pts), nodata_val)

            for i, (x, y) in enumerate(pts):
                row, col = rowcol(transform, x, y)
                if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
                    val = data[row, col]
                    elevations[i] = nodata_val if (nodata is not None and val == nodata) else val

            profile["elevations"] = elevations

    return profiles


def _tangent_at(line: LineString, distance: float, delta: float = 0.5) -> np.ndarray:
    """Bereken de genormaliseerde tangent-vector op een punt langs de lijn."""
    d0 = max(0, distance - delta)
    d1 = min(line.length, distance + delta)
    p0 = line.interpolate(d0)
    p1 = line.interpolate(d1)
    vec = np.array([p1.x - p0.x, p1.y - p0.y])
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return np.array([1.0, 0.0])
    return vec / norm
