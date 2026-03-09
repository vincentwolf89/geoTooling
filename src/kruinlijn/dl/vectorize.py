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

    # Sluit kleine gaten en verbind nabije componenten
    mask = binary_closing(mask, iterations=3)

    # Alle significante componenten verwerken (niet alleen de grootste)
    labeled, n = nd_label(mask)
    if n == 0:
        return None

    sizes = [0] + [(labeled == i).sum() for i in range(1, n + 1)]
    max_size = max(sizes[1:])
    min_component_size = max(50, max_size * 0.05)  # minimaal 5% van grootste

    all_lines = []
    for comp_id in range(1, n + 1):
        if sizes[comp_id] < min_component_size:
            continue

        comp_mask = labeled == comp_id
        skeleton = skeletonize(comp_mask)

        rows, cols = np.where(skeleton)
        if len(rows) < 5:
            continue

        xs = transform.c + cols * transform.a + 0.5 * transform.a
        ys = transform.f + rows * transform.e + 0.5 * transform.e

        coords = np.column_stack([xs, ys])
        ordered = _order_points(coords)

        if len(ordered) < 3:
            continue

        if smooth_sigma > 0:
            from scipy.ndimage import gaussian_filter1d
            sigma = min(smooth_sigma, len(ordered) / 5)
            ordered[:, 0] = gaussian_filter1d(ordered[:, 0], sigma=sigma, mode="nearest")
            ordered[:, 1] = gaussian_filter1d(ordered[:, 1], sigma=sigma, mode="nearest")

        comp_line = LineString(ordered)
        if simplify_tolerance > 0:
            comp_line = comp_line.simplify(simplify_tolerance, preserve_topology=True)

        if comp_line.length > 10:
            all_lines.append(comp_line)

    if not all_lines:
        return None

    # Merge alle componenten tot één lijn
    if len(all_lines) == 1:
        return all_lines[0]

    merged = linemerge(all_lines)
    if merged.geom_type == "LineString":
        return merged
    elif merged.geom_type == "MultiLineString":
        # Probeer nabije lijnen te verbinden (binnen 20m)
        connected = _connect_nearby_lines(list(merged.geoms), max_gap=20.0)
        if connected.geom_type == "LineString":
            return connected
        # Neem de langste als verbinden niet lukt
        return max(connected.geoms, key=lambda g: g.length)

    return max(all_lines, key=lambda g: g.length)


def _zone_edge_line(
    labels: np.ndarray,
    class_a: int,
    class_b: int,
    transform: rasterio.Affine,
    simplify_tolerance: float = 1.0,
    smooth_sigma: float = 3.0,
) -> LineString | None:
    """Extraheer de grenslijn tussen twee aangrenzende zones."""
    return _zone_edge_line_multi(
        labels, [class_a], [class_b], transform, simplify_tolerance, smooth_sigma
    )


def _zone_edge_line_multi(
    labels: np.ndarray,
    classes_a: list[int],
    classes_b: list[int],
    transform: rasterio.Affine,
    simplify_tolerance: float = 1.0,
    smooth_sigma: float = 3.0,
) -> LineString | None:
    """Extraheer de grenslijn tussen twee groepen klassen.

    Maakt een smal masker van pixels die op de grens liggen en skeletoniseert dat.
    """
    from scipy.ndimage import binary_dilation

    mask_a = np.isin(labels, classes_a)
    mask_b = np.isin(labels, classes_b)

    if mask_a.sum() < 10 or mask_b.sum() < 10:
        return None

    # Grenszone: pixels van groep_a die grenzen aan groep_b, en vice versa
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


def _connect_nearby_lines(
    lines: list, max_gap: float = 20.0
) -> "LineString | MultiLineString":
    """Verbind nabije lijnstukken door gaten op te vullen.

    Verbindt lijn-eindpunten die binnen max_gap meter van elkaar liggen.
    """
    if len(lines) <= 1:
        return lines[0] if lines else MultiLineString()

    # Sorteer op x van startpunt
    lines = sorted(lines, key=lambda l: l.coords[0][0])

    merged = [list(lines[0].coords)]
    for line in lines[1:]:
        last_end = merged[-1][-1]
        start = line.coords[0]
        end = line.coords[-1]

        dist_start = ((last_end[0] - start[0])**2 + (last_end[1] - start[1])**2)**0.5
        dist_end = ((last_end[0] - end[0])**2 + (last_end[1] - end[1])**2)**0.5

        if dist_start <= max_gap:
            merged[-1].extend(line.coords[1:])
        elif dist_end <= max_gap:
            merged[-1].extend(list(line.coords)[::-1][1:])
        else:
            merged.append(list(line.coords))

    result_lines = [LineString(c) for c in merged if len(c) >= 2]
    if len(result_lines) == 1:
        return result_lines[0]
    return MultiLineString(result_lines)


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

    Extraheert (indien aanwezig):
    - kruinlijn, binnenkruinlijn, buitenkruinlijn
    - binnenberm, buitenberm
    - binnenteenlijn, buitenteenlijn
    - insteeklijn, slootrand

    Parameters
    ----------
    prediction_path : str | Path
        Pad naar het voorspellings-GeoTIFF (uint8 labels 0-9).
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

    # Klassen: 1=kruin, 2=talud_binnen, 3=binnenberm, 4=teen_binnen,
    #          5=talud_buiten, 6=buitenberm, 7=teen_buiten, 8=insteek, 9=sloot

    def edge(a, b):
        return _zone_edge_line(labels, a, b, transform, simplify_tolerance, smooth_sigma)

    def edge_m(a_list, b_list):
        return _zone_edge_line_multi(labels, a_list, b_list, transform, simplify_tolerance, smooth_sigma)

    # Binnenkruinlijn: grens kruin-talud_binnen → fallback naar kruin-binnenberm
    # → uiterste fallback: grens kruin met gehele binnenzone
    lines["binnenkruinlijn"] = (
        edge(1, 2) or
        edge(1, 3) or
        edge_m([1], [2, 3, 4])
    )

    # Buitenkruinlijn: grens kruin-talud_buiten → fallback naar kruin-buitenberm
    # → uiterste fallback: grens kruin met gehele buitenzone
    lines["buitenkruinlijn"] = (
        edge(1, 5) or
        edge(1, 6) or
        edge_m([1], [5, 6, 7])
    )

    # Bermlijnen (middenlijnen van bermzones)
    lines["binnenberm"] = _class_centerline(
        labels, 3, transform, simplify_tolerance, smooth_sigma
    )
    lines["buitenberm"] = _class_centerline(
        labels, 6, transform, simplify_tolerance, smooth_sigma
    )

    # Binnenteenlijn: talud_binnen-teen → berm-teen → teen-achtergrond/insteek
    # → uiterste fallback: buitenrand van gehele binnenzone
    lines["binnenteenlijn"] = (
        edge(2, 4) or
        edge(3, 4) or
        edge(4, 0) or
        edge(4, 8) or
        edge_m([2, 3, 4], [0, 8, 9])
    )

    # Buitenteenlijn: talud_buiten-teen → berm-teen → teen-achtergrond/sloot
    # → uiterste fallback: buitenrand van gehele buitenzone
    lines["buitenteenlijn"] = (
        edge(5, 7) or
        edge(6, 7) or
        edge(7, 0) or
        edge(7, 9) or
        edge_m([5, 6, 7], [0, 8, 9])
    )

    # Insteeklijn
    lines["insteeklijn"] = _class_centerline(
        labels, 8, transform, simplify_tolerance, smooth_sigma
    )

    # Slootrand (grens water-land)
    lines["slootrand"] = _zone_edge_line(
        labels, 9, 0, transform, simplify_tolerance, smooth_sigma
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
