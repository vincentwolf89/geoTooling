"""Pipeline functies voor label-generatie op basis van referentielijnen."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import LineString, mapping


def generate_labels_from_lines(
    dtm_path: str | Path,
    reference_lines: dict[str, LineString | None],
    output_labels_path: str | Path,
    kruin_buffer: float = 2.0,
    teen_buffer: float = 2.5,
) -> np.ndarray:
    """Genereer segmentatie-labels op basis van echte referentielijnen.

    Maakt zone-polygonen tussen de referentielijnen (binnenkruin, buitenkruin,
    binnenteen, buitenteen) en rasterizeert deze als labels.

    Parameters
    ----------
    dtm_path : str | Path
        Pad naar het DTM GeoTIFF (EPSG:28992).
    reference_lines : dict
        Dict met keys 'binnenkruin', 'buitenkruin', 'binnenteen', 'buitenteen'.
        Waarden zijn Shapely LineStrings in EPSG:28992, of None.
    output_labels_path : str | Path
        Pad voor het output label-raster.
    kruin_buffer : float
        Buffer (m) rond kruinlijnen voor de kruinzone als fallback.
    teen_buffer : float
        Buffer (m) rond teenlijnen.

    Returns
    -------
    np.ndarray
        Label-array (0=achtergrond, 1=kruin, 2=talud_binnen, 3=teen_binnen,
        4=talud_buiten, 5=teen_buiten).
    """
    with rasterio.open(dtm_path) as src:
        raster_shape = (src.height, src.width)
        transform = src.transform
        profile = src.profile.copy()

    labels = np.zeros(raster_shape, dtype=np.uint8)

    binnenkruin = reference_lines.get("binnenkruin")
    buitenkruin = reference_lines.get("buitenkruin")
    binnenteen = reference_lines.get("binnenteen")
    buitenteen = reference_lines.get("buitenteen")

    shapes_to_rasterize = []

    # 1. Kruinzone: polygon tussen binnenkruin en buitenkruin
    if binnenkruin is not None and buitenkruin is not None:
        kruin_poly = _polygon_between_lines(binnenkruin, buitenkruin)
        if kruin_poly is not None and not kruin_poly.is_empty:
            kruin_poly = kruin_poly.buffer(0.5)
            shapes_to_rasterize.append((kruin_poly, 1))
    elif binnenkruin is not None:
        shapes_to_rasterize.append((binnenkruin.buffer(kruin_buffer), 1))
    elif buitenkruin is not None:
        shapes_to_rasterize.append((buitenkruin.buffer(kruin_buffer), 1))

    # 2. Talud binnen: polygon tussen binnenteen en binnenkruin
    if binnenteen is not None and binnenkruin is not None:
        talud_b_poly = _polygon_between_lines(binnenteen, binnenkruin)
        if talud_b_poly is not None and not talud_b_poly.is_empty:
            shapes_to_rasterize.append((talud_b_poly, 2))

    # 3. Talud buiten: polygon tussen buitenkruin en buitenteen
    if buitenkruin is not None and buitenteen is not None:
        talud_bu_poly = _polygon_between_lines(buitenkruin, buitenteen)
        if talud_bu_poly is not None and not talud_bu_poly.is_empty:
            shapes_to_rasterize.append((talud_bu_poly, 4))

    # 4. Teen zones: buffer rond teenlijnen
    if binnenteen is not None:
        shapes_to_rasterize.append((binnenteen.buffer(teen_buffer), 3))
    if buitenteen is not None:
        shapes_to_rasterize.append((buitenteen.buffer(teen_buffer), 5))

    # 5. Kruinlijnen zelf met buffer (overschrijft taludzones op kruingrens)
    if binnenkruin is not None:
        shapes_to_rasterize.append((binnenkruin.buffer(1.0), 1))
    if buitenkruin is not None:
        shapes_to_rasterize.append((buitenkruin.buffer(1.0), 1))

    if not shapes_to_rasterize:
        print("  Geen referentielijnen beschikbaar voor labels.")
        profile.update(dtype="uint8", count=1, nodata=0)
        with rasterio.open(output_labels_path, "w", **profile) as dst:
            dst.write(labels, 1)
        return labels

    # Rasterize: later in de lijst overschrijft eerder
    labels = rasterize(
        [(mapping(geom), value) for geom, value in shapes_to_rasterize],
        out_shape=raster_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    )

    # Schrijf label-raster
    profile.update(dtype="uint8", count=1, nodata=0)
    with rasterio.open(output_labels_path, "w", **profile) as dst:
        dst.write(labels, 1)

    # Statistieken
    for cls_id, cls_name in [(1, "kruin"), (2, "talud_binnen"), (3, "teen_binnen"),
                              (4, "talud_buiten"), (5, "teen_buiten")]:
        count = (labels == cls_id).sum()
        if count > 0:
            print(f"    {cls_name}: {count} pixels")

    return labels


def _polygon_between_lines(line1: LineString, line2: LineString) -> "Polygon | None":
    """Maak een polygon tussen twee roughly parallelle lijnen."""
    from shapely.geometry import Polygon, Point

    try:
        coords1 = list(line1.coords)
        coords2 = list(line2.coords)

        # Zorg dat lijnen in dezelfde richting lopen
        d_start = Point(coords2[0]).distance(Point(coords1[0]))
        d_end = Point(coords2[0]).distance(Point(coords1[-1]))
        if d_end < d_start:
            coords2 = list(reversed(coords2))

        ring = coords1 + list(reversed(coords2)) + [coords1[0]]
        poly = Polygon(ring)

        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return None

        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)

        return poly
    except Exception:
        return None
