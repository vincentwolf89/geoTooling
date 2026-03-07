"""Vectorisatie: extraheer kruinlijnen en teenlijnen uit DL segmentatiemasker."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge
import geopandas as gpd

from .dataset import CLASSES


def _mask_boundary_line(
    mask: np.ndarray,
    transform: rasterio.Affine,
    simplify_tolerance: float = 1.0,
    smooth_sigma: float = 3.0,
) -> LineString | None:
    """Extraheer de middenlijn van een binair masker via skeleton.

    Parameters
    ----------
    mask : np.ndarray
        Binair masker (True/False).
    transform : rasterio.Affine
        Geo-transform van het raster.
    simplify_tolerance : float
        Douglas-Peucker tolerance in meters.
    smooth_sigma : float
        Gaussian smoothing sigma voor de coördinaten.

    Returns
    -------
    LineString | None
        De geëxtraheerde lijn, of None als het masker te klein is.
    """
    from scipy.ndimage import binary_closing, label as nd_label
    from skimage.morphology import skeletonize

    if mask.sum() < 50:
        return None

    # Sluit kleine gaten
    mask = binary_closing(mask, iterations=2)

    # Grootste component behouden
    labeled, n = nd_label(mask)
    if n == 0:
        return None
    sizes = [0] + [(labeled == i).sum() for i in range(1, n + 1)]
    biggest = np.argmax(sizes)
    mask = labeled == biggest

    # Skeletonize naar 1px breed
    skeleton = skeletonize(mask)

    # Pixels naar coördinaten
    rows, cols = np.where(skeleton)
    if len(rows) < 5:
        return None

    # Pixel-coördinaten naar geo-coördinaten
    xs = transform.c + cols * transform.a + 0.5 * transform.a
    ys = transform.f + rows * transform.e + 0.5 * transform.e

    # Orden punten langs de lijn via nearest-neighbor chain
    coords = np.column_stack([xs, ys])
    ordered = _order_points(coords)

    if len(ordered) < 3:
        return None

    # Smooth de coördinaten
    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        sigma = min(smooth_sigma, len(ordered) / 5)
        ordered[:, 0] = gaussian_filter1d(ordered[:, 0], sigma=sigma, mode="nearest")
        ordered[:, 1] = gaussian_filter1d(ordered[:, 1], sigma=sigma, mode="nearest")

    line = LineString(ordered)

    # Simplify
    if simplify_tolerance > 0:
        line = line.simplify(simplify_tolerance, preserve_topology=True)

    return line if line.length > 10 else None


def _zone_edge_line(
    labels: np.ndarray,
    class_a: int,
    class_b: int,
    transform: rasterio.Affine,
    simplify_tolerance: float = 1.0,
    smooth_sigma: float = 3.0,
) -> LineString | None:
    """Extraheer de grenslijn tussen twee aangrenzende zones.

    Maakt een smal masker van pixels die op de grens liggen (class_a naast class_b)
    en skeletonizeert dat tot een lijn.
    """
    from scipy.ndimage import binary_dilation

    mask_a = labels == class_a
    mask_b = labels == class_b

    if mask_a.sum() < 10 or mask_b.sum() < 10:
        return None

    # Grenszone: pixels van class_a die grenzen aan class_b, en vice versa
    dilated_a = binary_dilation(mask_a, iterations=1)
    dilated_b = binary_dilation(mask_b, iterations=1)
    boundary = (dilated_a & mask_b) | (dilated_b & mask_a)

    return _mask_boundary_line(boundary, transform, simplify_tolerance, smooth_sigma)


def _class_centerline(
    labels: np.ndarray,
    class_id: int,
    transform: rasterio.Affine,
    simplify_tolerance: float = 1.0,
    smooth_sigma: float = 3.0,
) -> LineString | None:
    """Extraheer de middenlijn van een zone via skeletonize."""
    mask = labels == class_id
    return _mask_boundary_line(mask, transform, simplify_tolerance, smooth_sigma)


def _order_points(coords: np.ndarray) -> np.ndarray:
    """Orden ongeordende 2D punten via nearest-neighbor chain.

    Start bij het punt met de laagste x-coördinaat.
    """
    from scipy.spatial import cKDTree

    n = len(coords)
    if n < 3:
        return coords

    tree = cKDTree(coords)
    visited = np.zeros(n, dtype=bool)

    # Start bij punt met laagste x
    start = np.argmin(coords[:, 0])
    order = [start]
    visited[start] = True

    current = start
    for _ in range(n - 1):
        dists, indices = tree.query(coords[current], k=min(20, n))
        found = False
        for d, idx in zip(dists, indices):
            if not visited[idx]:
                order.append(idx)
                visited[idx] = True
                current = idx
                found = True
                break
        if not found:
            break

    return coords[order]


def extract_lines(
    prediction_path: str | Path,
    output_gpkg: str | Path | None = None,
    simplify_tolerance: float = 1.5,
    smooth_sigma: float = 5.0,
) -> dict[str, LineString | None]:
    """Extraheer vector-lijnen uit een DL voorspellings-raster.

    Extraheert:
    - kruinlijn: middenlijn van de kruinzone (klasse 1)
    - binnenkruinlijn: grens tussen kruin (1) en talud_binnen (2)
    - buitenkruinlijn: grens tussen kruin (1) en talud_buiten (4)
    - binnenteenlijn: grens tussen talud_binnen (2) en teen_binnen (3)
    - buitenteenlijn: grens tussen talud_buiten (4) en teen_buiten (5)

    Parameters
    ----------
    prediction_path : str | Path
        Pad naar het voorspellings-GeoTIFF (uint8 labels 0-5).
    output_gpkg : str | Path | None
        Optioneel pad om resultaten als GeoPackage op te slaan.
    simplify_tolerance : float
        Douglas-Peucker simplify tolerance in meters.
    smooth_sigma : float
        Gaussian smoothing sigma voor lijncoördinaten.

    Returns
    -------
    dict[str, LineString | None]
        Dict met lijnnamen als keys en LineStrings als values.
    """
    with rasterio.open(prediction_path) as src:
        labels = src.read(1)
        transform = src.transform
        crs = src.crs

    lines = {}

    # Kruinlijn: middenlijn van de kruinzone
    lines["kruinlijn"] = _class_centerline(
        labels, 1, transform, simplify_tolerance, smooth_sigma
    )

    # Grenslijnen tussen zones
    lines["binnenkruinlijn"] = _zone_edge_line(
        labels, 1, 2, transform, simplify_tolerance, smooth_sigma
    )
    lines["buitenkruinlijn"] = _zone_edge_line(
        labels, 1, 4, transform, simplify_tolerance, smooth_sigma
    )
    lines["binnenteenlijn"] = _zone_edge_line(
        labels, 2, 3, transform, simplify_tolerance, smooth_sigma
    )
    lines["buitenteenlijn"] = _zone_edge_line(
        labels, 4, 5, transform, simplify_tolerance, smooth_sigma
    )

    # Samenvatting
    for name, line in lines.items():
        if line is not None:
            print(f"  {name}: {line.length:.0f}m ({len(line.coords)} punten)")
        else:
            print(f"  {name}: niet gevonden")

    # Opslaan als GeoPackage
    if output_gpkg is not None:
        output_gpkg = Path(output_gpkg)
        output_gpkg.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for name, line in lines.items():
            if line is not None:
                rows.append({"naam": name, "lengte_m": round(line.length, 1), "geometry": line})

        if rows:
            gdf = gpd.GeoDataFrame(rows, crs=crs)
            gdf.to_file(output_gpkg, driver="GPKG")
            print(f"Lijnen opgeslagen: {output_gpkg}")

    return lines
