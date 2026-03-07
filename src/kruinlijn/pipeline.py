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
    spacing: float = 2.0,
) -> np.ndarray:
    """Genereer segmentatie-labels via per-profiel knikpunt-gebaseerde labeling.

    Loopt langs de hartlijn met dichte profielen. Per profiel worden pixels
    gelabeld op basis van hun positie t.o.v. de gedetecteerde knikpunten
    (binnenteen, binnenkruin, buitenkruin, buitenteen). Dit geeft veel
    preciezere labels dan buffer-gebaseerde zones.

    Parameters
    ----------
    dtm_path : str | Path
        Pad naar het DTM GeoTIFF.
    centerline : LineString
        Ruwe hartlijn van de dijk.
    output_labels_path : str | Path
        Pad voor het output label-raster.
    crest_buffer : float
        Halve breedte (m) van de kruinzone.
    talud_width : float
        Fallback breedte (m) als een kniklijn niet gedetecteerd is.
    teen_buffer : float
        Halve breedte (m) van de teenzone.
    water_side : str | None
        'left' of 'right' om buitenzijde te forceren.
    spacing : float
        Afstand (m) tussen profielen voor labeling.

    Returns
    -------
    np.ndarray
        Label-array (0=achtergrond, 1=kruin, 2=talud_binnen, 3=teen_binnen,
        4=talud_buiten, 5=teen_buiten).
    """
    # Stap 1: detecteer knikpunten op profielen
    profiles = generate_cross_profiles(centerline, spacing=spacing, width=60.0)
    profiles = extract_profile_elevations(profiles, str(dtm_path))
    profiles = detect_crest_points(profiles, smooth_sigma=2.0, method="curvature")
    profiles = detect_knikpunten(profiles, smooth_sigma=3.0, water_side=water_side)

    with rasterio.open(dtm_path) as src:
        raster_shape = (src.height, src.width)
        transform = src.transform
        profile = src.profile.copy()

    labels = np.zeros(raster_shape, dtype=np.uint8)

    # Stap 2: per profiel, label pixels via vectorized coords
    inv_a = 1.0 / transform.a
    inv_e = 1.0 / transform.e

    for p in profiles:
        crest_idx = p.get("crest_idx")
        if crest_idx is None:
            continue

        offsets = p["offsets"]
        line = p["line"]
        n_pts = len(offsets)

        # Haal profiel-coördinaten direct uit de LineString (veel sneller)
        coords = np.array(line.coords)
        if len(coords) != n_pts:
            # Interpoleer indien nodig
            fracs = np.linspace(0, 1, n_pts)
            xs = np.interp(fracs, np.linspace(0, 1, len(coords)), coords[:, 0])
            ys = np.interp(fracs, np.linspace(0, 1, len(coords)), coords[:, 1])
        else:
            xs, ys = coords[:, 0], coords[:, 1]

        # Vectorized pixel-coordinaten
        cols = np.round((xs - transform.c) * inv_a).astype(int)
        rows = np.round((ys - transform.f) * inv_e).astype(int)

        # Knikpunt-indices
        bk_idx = p.get("binnenkruin_idx")
        buk_idx = p.get("buitenkruin_idx")
        bt_idx = p.get("binnenteen_idx")
        but_idx = p.get("buitenteen_idx")

        # Classificeer alle punten vectorized
        for i in range(n_pts):
            r, c = rows[i], cols[i]
            if r < 0 or r >= raster_shape[0] or c < 0 or c >= raster_shape[1]:
                continue

            label = _classify_profile_point(
                i, crest_idx, bk_idx, buk_idx, bt_idx, but_idx,
                crest_buffer, teen_buffer, offsets,
            )
            if label > 0:
                labels[r, c] = label

    # Stap 3: morfologische dilation om gaten te vullen
    from scipy.ndimage import maximum_filter
    n_dilate = max(2, int(spacing / 0.5))
    for _ in range(n_dilate):
        dilated = maximum_filter(labels, size=3)
        labels = np.where(labels == 0, dilated, labels)

    # Stap 4: schrijf label-raster
    profile.update(dtype="uint8", count=1, nodata=0)
    with rasterio.open(output_labels_path, "w", **profile) as dst:
        dst.write(labels, 1)

    print(f"Labels gegenereerd: {output_labels_path}")
    return labels


def _classify_profile_point(
    i: int, crest_idx: int,
    bk_idx: int | None, buk_idx: int | None,
    bt_idx: int | None, but_idx: int | None,
    crest_half: float, teen_half: float,
    offsets: np.ndarray,
) -> int:
    """Classificeer een profiel-punt op basis van knikpunt-posities.

    Returns label: 0=achtergrond, 1=kruin, 2=talud_binnen, 3=teen_binnen,
    4=talud_buiten, 5=teen_buiten.
    """
    offset = offsets[i]
    crest_offset = offsets[crest_idx]

    # Bepaal kruinzone-grenzen
    kruin_left = offsets[bk_idx] if bk_idx is not None else crest_offset - crest_half
    kruin_right = offsets[buk_idx] if buk_idx is not None else crest_offset + crest_half
    if kruin_left > kruin_right:
        kruin_left, kruin_right = kruin_right, kruin_left

    # Kruin
    if kruin_left - 0.5 <= offset <= kruin_right + 0.5:
        return 1

    # Binnenzijde (links van kruin in offset-ruimte, maar dit is al correct
    # want detect_knikpunten plaatst binnen-punten links van crest)
    if offset < kruin_left:
        # Teen binnen
        if bt_idx is not None:
            bt_offset = offsets[bt_idx]
            if abs(offset - bt_offset) <= teen_half:
                return 3
            # Talud binnen: tussen teen en kruinrand
            if bt_offset <= offset < kruin_left:
                return 2
            # Voorbij de teen = achtergrond
            return 0
        else:
            # Geen teen gedetecteerd: alles tot max talud_width is talud
            return 2 if offset >= kruin_left - 15.0 else 0

    # Buitenzijde (rechts van kruin)
    if offset > kruin_right:
        if but_idx is not None:
            but_offset = offsets[but_idx]
            if abs(offset - but_offset) <= teen_half:
                return 5
            # Talud buiten: tussen kruinrand en teen
            if kruin_right < offset <= but_offset:
                return 4
            return 0
        else:
            return 4 if offset <= kruin_right + 15.0 else 0

    return 0
