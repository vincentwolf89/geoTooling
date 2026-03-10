"""Post-processing: merge kniklijnen uit overlappende secties tot doorlopende lijnen.

Simpele aanpak: snap nabije eindpunten en linemerge. Geen interpolatie.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, snap, unary_union


TARGET_LINES = [
    "binnenkruinlijn",
    "buitenkruinlijn",
    "binnenteenlijn",
    "buitenteenlijn",
]


def _explode_to_lines(geom) -> list[LineString]:
    """Splits een geometry in individuele LineStrings."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom]
    elif geom.geom_type == "MultiLineString":
        return [g for g in geom.geoms if g.length > 5]
    return []


def _snap_and_merge(lines: list[LineString], snap_tolerance: float = 50.0) -> LineString | MultiLineString | None:
    """Snap nabije eindpunten en merge lijnfragmenten.

    Stap 1: Snap elk eindpunt naar het dichtstbijzijnde eindpunt van een andere lijn
    Stap 2: linemerge om aaneengesloten stukken samen te voegen
    Stap 3: Filter korte fragmenten (<50m)
    """
    if not lines:
        return None
    if len(lines) == 1:
        return lines[0]

    # Verzamel alle eindpunten
    endpoints = []
    for i, line in enumerate(lines):
        endpoints.append((i, "start", Point(line.coords[0])))
        endpoints.append((i, "end", Point(line.coords[-1])))

    # Snap: voor elk eindpunt, zoek het dichtstbijzijnde eindpunt van een ANDERE lijn
    snapped_lines = list(lines)
    for i, line in enumerate(lines):
        coords = list(line.coords)
        for end_idx, coord_idx in [("start", 0), ("end", -1)]:
            pt = Point(coords[coord_idx])
            best_dist = snap_tolerance
            best_target = None
            for j, _, other_pt in endpoints:
                if j == i:
                    continue
                d = pt.distance(other_pt)
                if d < best_dist:
                    best_dist = d
                    best_target = other_pt
            if best_target is not None:
                coords[coord_idx] = (best_target.x, best_target.y)
        snapped_lines[i] = LineString(coords)

    # linemerge
    merged = linemerge(snapped_lines)

    # Filter korte fragmenten
    if merged.geom_type == "MultiLineString":
        significant = [g for g in merged.geoms if g.length > 50]
        if not significant:
            return None
        if len(significant) == 1:
            return significant[0]
        return MultiLineString(significant)

    return merged


def _deduplicate_overlapping(
    lines: list[LineString],
    trajectory: LineString,
    overlap_tolerance: float = 30.0,
) -> list[LineString]:
    """Verwijder overlappende fragmenten uit overlappende secties.

    Secties overlappen 200m, dus dezelfde dijkstukken worden dubbel gevectoriseerd.
    Houd per stuk het langste/beste fragment.
    """
    if len(lines) <= 1:
        return lines

    # Zorg dat traject 2D is
    if trajectory.has_z:
        trajectory = LineString([(x, y) for x, y, *_ in trajectory.coords])

    # Projecteer elk fragment op het traject
    projected = []
    for line in lines:
        mid = line.interpolate(0.5, normalized=True)
        proj = trajectory.project(Point(mid.x, mid.y))
        start = trajectory.project(Point(line.coords[0]))
        end = trajectory.project(Point(line.coords[-1]))
        if start > end:
            start, end = end, start
        projected.append({
            "line": line,
            "start": start,
            "end": end,
            "mid": proj,
            "length": line.length,
        })

    # Sorteer op startpositie
    projected.sort(key=lambda f: f["start"])

    # Greedy: houd het langste fragment per overlappend stuk
    kept = [projected[0]]
    for frag in projected[1:]:
        prev = kept[-1]
        # Check overlap: als het midden van frag binnen het bereik van prev valt
        overlap = min(prev["end"], frag["end"]) - max(prev["start"], frag["start"])
        min_len = min(prev["length"], frag["length"])

        if overlap > min_len * 0.5:
            # Significante overlap: houd de langste
            if frag["length"] > prev["length"]:
                kept[-1] = frag
        else:
            kept.append(frag)

    return [f["line"] for f in kept]


def merge_sections(
    predict_dir: str | Path,
    traject: LineString,
    output_gpkg: str | Path,
    snap_tolerance: float = 50.0,
    line_names: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Merge kniklijnen uit alle secties van een traject.

    Parameters
    ----------
    predict_dir : Path
        Map met section_001/, section_002/, etc.
    traject : LineString
        Het traject (in RD) waarlangs de secties liggen.
    output_gpkg : Path
        Output GeoPackage pad.
    snap_tolerance : float
        Maximale afstand (m) voor het snappen van eindpunten.
    line_names : list[str]
        Welke lijntypes samenvoegen.
    """
    predict_dir = Path(predict_dir)
    if line_names is None:
        line_names = TARGET_LINES + ["insteeklijn", "slootrand"]

    # Verzamel alle fragmenten per lijntype
    all_fragments: dict[str, list[LineString]] = {name: [] for name in line_names}
    crs = None

    for section_dir in sorted(predict_dir.glob("section_*")):
        gpkg = section_dir / "kniklijnen.gpkg"
        if not gpkg.exists():
            continue
        gdf = gpd.read_file(gpkg)
        if crs is None:
            crs = gdf.crs

        for _, row in gdf.iterrows():
            naam = row.get("naam", "")
            if naam in all_fragments:
                all_fragments[naam].extend(_explode_to_lines(row.geometry))

    # Merge per lijntype
    rows = []
    for naam, fragments in all_fragments.items():
        if not fragments:
            print(f"  {naam}: geen fragmenten")
            continue

        total_len = sum(f.length for f in fragments)
        print(f"  {naam}: {len(fragments)} fragmenten, {total_len:.0f}m")

        # Stap 1: deduplicate overlappende fragmenten
        deduped = _deduplicate_overlapping(fragments, traject)
        if len(deduped) < len(fragments):
            dedup_len = sum(f.length for f in deduped)
            print(f"    dedup: {len(fragments)} -> {len(deduped)} ({dedup_len:.0f}m)")

        # Stap 2: snap en merge
        merged = _snap_and_merge(deduped, snap_tolerance=snap_tolerance)

        if merged is not None:
            n_parts = len(merged.geoms) if merged.geom_type == "MultiLineString" else 1
            print(f"    -> {merged.length:.0f}m ({n_parts} {'deel' if n_parts == 1 else 'delen'})")
            rows.append({
                "naam": naam,
                "lengte_m": round(merged.length, 1),
                "geometry": merged,
            })

    if not rows:
        print("  Geen lijnen gevonden!")
        return gpd.GeoDataFrame()

    result = gpd.GeoDataFrame(rows, crs=crs)
    output_gpkg = Path(output_gpkg)
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    result.to_file(output_gpkg, driver="GPKG")
    print(f"  Opgeslagen: {output_gpkg}")
    return result
