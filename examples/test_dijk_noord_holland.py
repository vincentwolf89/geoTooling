"""Test: kruinlijndetectie voor een stukje dijk in Noord-Holland.

Genereert een synthetisch DTM (gebaseerd op een realistisch dijkprofiel)
voor een locatie nabij de Markermeerdijk bij Uitdam, Noord-Holland.
Draait vervolgens de morfologische kruinlijndetectie pipeline.

Gebruik:
    python examples/test_dijk_noord_holland.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_bounds
from rasterio.plot import show
from shapely.geometry import LineString

# Voeg src toe aan het pad
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kruinlijn.pipeline import morpho_pipeline


def maak_synthetisch_dijkprofiel(x_offset: float, variatie: float = 0.0) -> callable:
    """Geeft een functie die de hoogte (m NAP) berekent voor een dwarsprofiel.

    Simuleert een typisch dijkprofiel:
    - Polder (binnenzijde): ca. -1.5 m NAP
    - Binnenteen: geleidelijke stijging
    - Binnentalud: steil omhoog
    - Kruin: ca. 4.5 m NAP (met lichte variatie)
    - Buitentalud: steil omlaag richting water
    - Buitenteen: overgang naar voorland/water
    - Voorland: ca. 0 m NAP
    """
    kruin_hoogte = 4.5 + variatie

    def profiel(afstand_dwars: float) -> float:
        """afstand_dwars: afstand in m loodrecht op hartlijn, negatief = polder, positief = water."""
        d = afstand_dwars
        if d < -15:
            return -1.5  # polder
        elif d < -8:
            # binnenteen naar binnentalud
            t = (d + 15) / 7.0
            return -1.5 + t * 1.5
        elif d < -2:
            # binnentalud
            t = (d + 8) / 6.0
            return 0.0 + t * kruin_hoogte
        elif d < 2:
            # kruin (vlak)
            return kruin_hoogte
        elif d < 8:
            # buitentalud
            t = (d - 2) / 6.0
            return kruin_hoogte - t * (kruin_hoogte - 0.5)
        elif d < 15:
            # buitenteen
            t = (d - 8) / 7.0
            return 0.5 - t * 0.5
        else:
            return 0.0  # voorland/water

    return profiel


def genereer_dtm(
    dtm_pad: Path,
    hartlijn: LineString,
    breedte: float = 50.0,
    resolutie: float = 0.5,
) -> None:
    """Genereer een synthetisch DTM GeoTIFF rond een hartlijn.

    Parameters
    ----------
    dtm_pad : Path
        Uitvoerpad voor het GeoTIFF.
    hartlijn : LineString
        Hartlijn van de dijk (in RD New / EPSG:28992).
    breedte : float
        Totale breedte van het DTM rond de hartlijn (m).
    resolutie : float
        Pixelgrootte in meters.
    """
    # Bepaal de bounding box
    minx, miny, maxx, maxy = hartlijn.buffer(breedte / 2 + 5).bounds

    # Maak raster
    ncols = int((maxx - minx) / resolutie)
    nrows = int((maxy - miny) / resolutie)
    transform = from_bounds(minx, miny, maxx, maxy, ncols, nrows)

    data = np.full((nrows, ncols), -1.5, dtype=np.float32)

    # Vul het raster per pixel
    np.random.seed(42)
    for row in range(nrows):
        for col in range(ncols):
            # Pixel-coordinaten naar wereld
            x = minx + col * resolutie + resolutie / 2
            y = maxy - row * resolutie - resolutie / 2

            from shapely.geometry import Point

            pt = Point(x, y)
            # Afstand tot hartlijn (positief = rechts, negatief = links)
            afstand = hartlijn.distance(pt)
            # Bepaal zijde (signed distance)
            nearest_pt = hartlijn.interpolate(hartlijn.project(pt))
            proj_dist = hartlijn.project(pt)

            # Bepaal normaal-richting op dit punt
            delta = 0.5
            d0 = max(0, proj_dist - delta)
            d1 = min(hartlijn.length, proj_dist + delta)
            p0 = hartlijn.interpolate(d0)
            p1 = hartlijn.interpolate(d1)
            tangent = np.array([p1.x - p0.x, p1.y - p0.y])
            norm = np.linalg.norm(tangent)
            if norm > 1e-10:
                tangent /= norm
            normal = np.array([-tangent[1], tangent[0]])

            # Signed distance: positief = richting normaal (waterzijde)
            vec = np.array([x - nearest_pt.x, y - nearest_pt.y])
            signed_dist = np.dot(vec, normal)

            # Variatie langs de dijk
            variatie = 0.3 * np.sin(proj_dist / 50.0)

            profiel_fn = maak_synthetisch_dijkprofiel(proj_dist, variatie)
            hoogte = profiel_fn(signed_dist)

            # Voeg wat ruis toe (realistisch terrein)
            hoogte += np.random.normal(0, 0.05)
            data[row, col] = hoogte

    # Schrijf GeoTIFF in RD New (EPSG:28992)
    with rasterio.open(
        dtm_pad,
        "w",
        driver="GTiff",
        height=nrows,
        width=ncols,
        count=1,
        dtype="float32",
        crs="EPSG:28992",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)

    print(f"DTM gegenereerd: {dtm_pad} ({ncols}x{nrows} pixels, {resolutie}m resolutie)")


def main():
    output_dir = Path("output/test_noord_holland")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Definieer een stukje hartlijn nabij Markermeerdijk, Uitdam (Noord-Holland)
    # Coordinaten in RD New (EPSG:28992)
    # Dit is een licht gebogen dijksectie van ~200m lang
    hartlijn = LineString([
        (133500, 490200),
        (133520, 490250),
        (133545, 490300),
        (133575, 490345),
        (133610, 490385),
        (133650, 490420),
        (133695, 490450),
    ])

    print("=" * 60)
    print("Test Kruinlijndetectie - Dijk Noord-Holland")
    print("Locatie: nabij Markermeerdijk, Uitdam")
    print(f"Dijklengte: {hartlijn.length:.0f} m")
    print("=" * 60)

    # Stap 1: Genereer synthetisch DTM
    dtm_pad = output_dir / "dtm_test_noordholland.tif"
    print("\n[1/3] Genereren synthetisch DTM...")
    genereer_dtm(dtm_pad, hartlijn, breedte=50.0, resolutie=0.5)

    # Stap 2: Draai de kruinlijndetectie pipeline
    print("\n[2/3] Uitvoeren kruinlijndetectie...")
    crest_line, points_gdf = morpho_pipeline(
        dtm_path=str(dtm_pad),
        centerline=hartlijn,
        output_gpkg=str(output_dir / "kruinlijn_resultaat.gpkg"),
        spacing=5.0,
        width=40.0,
        smooth_sigma=2.0,
        method="curvature",
    )

    # Resultaten
    print(f"\nResultaten:")
    print(f"  Kruinlijn gedetecteerd: {'Ja' if crest_line is not None else 'Nee'}")
    print(f"  Aantal kruinpunten: {len(points_gdf)}")
    if len(points_gdf) > 0:
        print(f"  Gemiddelde kruinhoogte: {points_gdf['z'].mean():.2f} m NAP")
        print(f"  Min kruinhoogte: {points_gdf['z'].min():.2f} m NAP")
        print(f"  Max kruinhoogte: {points_gdf['z'].max():.2f} m NAP")

    # Stap 3: Visualisatie
    print("\n[3/3] Maken visualisatie...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Links: DTM met kruinlijn
    with rasterio.open(str(dtm_pad)) as src:
        show(src, ax=axes[0], cmap="terrain", title="DTM met kruinlijn - Noord-Holland test")
    if crest_line is not None:
        x, y = crest_line.xy
        axes[0].plot(x, y, "r-", linewidth=2, label="Kruinlijn (gedetecteerd)")
    cx, cy = hartlijn.xy
    axes[0].plot(cx, cy, "b--", linewidth=1, alpha=0.5, label="Hartlijn (invoer)")
    axes[0].legend()
    axes[0].set_xlabel("X (RD)")
    axes[0].set_ylabel("Y (RD)")

    # Rechts: een voorbeeld dwarsprofiel
    from kruinlijn.morpho import generate_cross_profiles, extract_profile_elevations, detect_crest_points

    profiles = generate_cross_profiles(hartlijn, spacing=5.0, width=40.0)
    profiles = extract_profile_elevations(profiles, str(dtm_pad))
    profiles = detect_crest_points(profiles, method="curvature")

    mid = len(profiles) // 2
    p = profiles[mid]
    axes[1].plot(p["offsets"], p["elevations"], "k-", linewidth=1.5, label="Hoogteprofiel")
    if p["crest_idx"] is not None:
        axes[1].axvline(p["offsets"][p["crest_idx"]], color="r", linestyle="--", alpha=0.7, label="Kruin")
        axes[1].plot(p["offsets"][p["crest_idx"]], p["crest_z"], "ro", markersize=10, zorder=5)
    axes[1].set_xlabel("Afstand op dwarsprofiel (m)")
    axes[1].set_ylabel("Hoogte (m NAP)")
    axes[1].set_title(f"Dwarsprofiel op {p['distance']:.0f}m langs de dijk")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_pad = output_dir / "kruinlijn_analyse_noordholland.png"
    plt.savefig(plot_pad, dpi=150)
    print(f"  Plot opgeslagen: {plot_pad}")

    print("\n" + "=" * 60)
    print("Test voltooid!")
    print(f"Uitvoer in: {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
