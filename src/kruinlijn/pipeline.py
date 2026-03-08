"""Pipeline functies voor label-generatie op basis van referentielijnen en BGT."""

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
    berm_buffer: float = 2.0,
    insteek_buffer: float = 1.5,
    sloot_polygons: list | None = None,
) -> np.ndarray:
    """Genereer segmentatie-labels op basis van referentielijnen + BGT.

    Parameters
    ----------
    dtm_path : str | Path
        Pad naar het DTM GeoTIFF (EPSG:28992).
    reference_lines : dict
        Dict met keys: 'binnenkruin', 'buitenkruin', 'binnenteen', 'buitenteen',
        'binnenberm', 'buitenberm', 'insteek'. Waarden: LineStrings of None.
    output_labels_path : str | Path
        Pad voor het output label-raster.
    kruin_buffer : float
        Buffer (m) rond kruinlijnen als fallback.
    teen_buffer : float
        Buffer (m) rond teenlijnen.
    berm_buffer : float
        Buffer (m) rond bermlijnen.
    insteek_buffer : float
        Buffer (m) rond insteeklijn.
    sloot_polygons : list | None
        Lijst van Shapely Polygons voor sloten (bijv. uit BGT waterdeel).

    Returns
    -------
    np.ndarray
        Label-array: 0=achtergrond, 1=kruin, 2=talud_binnen, 3=binnenberm,
        4=teen_binnen, 5=talud_buiten, 6=buitenberm, 7=teen_buiten,
        8=insteek, 9=sloot.
    """
    with rasterio.open(dtm_path) as src:
        raster_shape = (src.height, src.width)
        transform = src.transform
        profile = src.profile.copy()

    binnenkruin = reference_lines.get("binnenkruin")
    buitenkruin = reference_lines.get("buitenkruin")
    binnenteen = reference_lines.get("binnenteen")
    buitenteen = reference_lines.get("buitenteen")
    binnenberm = reference_lines.get("binnenberm")
    buitenberm = reference_lines.get("buitenberm")
    insteek = reference_lines.get("insteek")

    # Volgorde van rasterize: later overschrijft eerder
    shapes = []

    # 1. Talud binnen: polygon tussen binnenteen en binnenkruin
    if binnenteen is not None and binnenkruin is not None:
        poly = _polygon_between_lines(binnenteen, binnenkruin)
        if poly is not None and not poly.is_empty:
            shapes.append((poly, 2))  # talud_binnen

    # 2. Talud buiten: polygon tussen buitenkruin en buitenteen
    if buitenkruin is not None and buitenteen is not None:
        poly = _polygon_between_lines(buitenkruin, buitenteen)
        if poly is not None and not poly.is_empty:
            shapes.append((poly, 5))  # talud_buiten

    # 3. Teen zones: buffer rond teenlijnen
    if binnenteen is not None:
        shapes.append((binnenteen.buffer(teen_buffer), 4))  # teen_binnen
    if buitenteen is not None:
        shapes.append((buitenteen.buffer(teen_buffer), 7))  # teen_buiten

    # 4. Bermen: buffer rond bermlijnen (overschrijft talud)
    if binnenberm is not None:
        shapes.append((binnenberm.buffer(berm_buffer), 3))  # binnenberm
    if buitenberm is not None:
        shapes.append((buitenberm.buffer(berm_buffer), 6))  # buitenberm

    # 5. Kruinzone: polygon tussen binnenkruin en buitenkruin
    if binnenkruin is not None and buitenkruin is not None:
        kruin_poly = _polygon_between_lines(binnenkruin, buitenkruin)
        if kruin_poly is not None and not kruin_poly.is_empty:
            kruin_poly = kruin_poly.buffer(0.5)
            shapes.append((kruin_poly, 1))  # kruin
    elif binnenkruin is not None:
        shapes.append((binnenkruin.buffer(kruin_buffer), 1))
    elif buitenkruin is not None:
        shapes.append((buitenkruin.buffer(kruin_buffer), 1))

    # 6. Kruinlijnen zelf (overschrijft alles behalve sloot)
    if binnenkruin is not None:
        shapes.append((binnenkruin.buffer(1.0), 1))
    if buitenkruin is not None:
        shapes.append((buitenkruin.buffer(1.0), 1))

    # 7. Insteek: expliciet meegegeven of afgeleid van sloot-randen
    if insteek is not None:
        shapes.append((insteek.buffer(insteek_buffer), 8))  # insteek

    # 8. Sloot + insteek uit BGT waterdeel polygonen
    # Alleen waterdelen die dicht bij de dijk liggen (binnen max_water_dist)
    if sloot_polygons:
        max_water_dist = 40.0  # meter van dijk-referentielijnen

        # Maak een dijkzone-lijn om afstand te berekenen
        dijk_lines = [l for l in [binnenteen, buitenteen, binnenkruin, buitenkruin] if l is not None]
        from shapely.ops import unary_union
        dijk_union = unary_union(dijk_lines) if dijk_lines else None

        for poly in sloot_polygons:
            if hasattr(poly, "geometry"):
                poly = poly["geometry"] if isinstance(poly, dict) else poly.geometry
            if isinstance(poly, dict):
                from shapely.geometry import shape as _shape
                poly = _shape(poly)
            if poly.is_empty:
                continue

            # Filter: alleen waterdelen dicht bij de dijk
            if dijk_union is not None and poly.distance(dijk_union) > max_water_dist:
                continue

            # Insteek: smalle buffer rond de rand van het waterdeel
            boundary = poly.boundary
            if not boundary.is_empty:
                shapes.append((boundary.buffer(insteek_buffer), 8))  # insteek
            # Sloot: het watervlak zelf (overschrijft insteek in het midden)
            shapes.append((poly, 9))  # sloot

    if not shapes:
        print("  Geen referentielijnen beschikbaar voor labels.")
        labels = np.zeros(raster_shape, dtype=np.uint8)
        profile.update(dtype="uint8", count=1, nodata=0)
        with rasterio.open(output_labels_path, "w", **profile) as dst:
            dst.write(labels, 1)
        return labels

    # Rasterize: later in de lijst overschrijft eerder
    labels = rasterize(
        [(mapping(geom), value) for geom, value in shapes],
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
    class_names = {
        1: "kruin", 2: "talud_binnen", 3: "binnenberm", 4: "teen_binnen",
        5: "talud_buiten", 6: "buitenberm", 7: "teen_buiten",
        8: "insteek", 9: "sloot",
    }
    for cls_id, cls_name in class_names.items():
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
