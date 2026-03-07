"""Detecteer kruinpunten en knikpunten op dwarsprofielen via morfologische analyse."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from shapely.geometry import LineString, Point

# Knikpunt-types die gedetecteerd worden
KNIKPUNT_TYPES = [
    "binnenteen",       # overgang polder → binnentalud
    "kruinrand_binnen", # overgang binnentalud → kruin
    "kruinrand_buiten", # overgang kruin → buitentalud
    "buitenteen",       # overgang buitentalud → voorland
]


def detect_crest_points(
    profiles: list[dict],
    smooth_sigma: float = 2.0,
    min_prominence: float = 0.3,
    method: str = "curvature",
) -> list[dict]:
    """Detecteer het kruinpunt per dwarsprofiel.

    Parameters
    ----------
    profiles : list[dict]
        Profielen met 'elevations', 'offsets' en 'points' keys.
    smooth_sigma : float
        Sigma voor Gaussische smoothing van het profiel.
    min_prominence : float
        Minimale prominentie (m) voor peak-detectie.
    method : str
        'peak' — vind de hoogste prominente piek.
        'curvature' — vind het punt met maximale negatieve kromming (convex top).

    Returns
    -------
    list[dict]
        Profielen aangevuld met:
        - 'crest_idx': index van het kruinpunt in het profiel (of None)
        - 'crest_xy': (x, y) coordinaat van het kruinpunt (of None)
        - 'crest_z': hoogte van het kruinpunt (of None)
    """
    for profile in profiles:
        z = profile["elevations"]
        pts = profile["points"]

        valid = np.isfinite(z)
        if valid.sum() < 5:
            profile.update({"crest_idx": None, "crest_xy": None, "crest_z": None})
            continue

        # Interpoleer NaN-gaten
        z_clean = _interpolate_nans(z.copy())
        z_smooth = gaussian_filter1d(z_clean, sigma=smooth_sigma)

        if method == "peak":
            idx = _detect_peak(z_smooth, min_prominence)
        elif method == "curvature":
            idx = _detect_curvature(z_smooth, min_prominence)
        else:
            raise ValueError(f"Onbekende methode: {method}")

        if idx is not None:
            profile["crest_idx"] = idx
            profile["crest_xy"] = tuple(pts[idx])
            profile["crest_z"] = z[idx]
        else:
            profile.update({"crest_idx": None, "crest_xy": None, "crest_z": None})

    return profiles


def detect_knikpunten(
    profiles: list[dict],
    smooth_sigma: float = 3.0,
) -> list[dict]:
    """Detecteer alle knikpunten (breeklijnen) per dwarsprofiel.

    Knikpunten zijn locaties waar de helling significant verandert:
    - binnenteen: overgang polder → binnentalud (positieve krommingspiek, binnenzijde)
    - kruinrand_binnen: overgang binnentalud → kruin (negatieve krommingspiek, binnenzijde)
    - kruinrand_buiten: overgang kruin → buitentalud (negatieve krommingspiek, buitenzijde)
    - buitenteen: overgang buitentalud → voorland (positieve krommingspiek, buitenzijde)

    Parameters
    ----------
    profiles : list[dict]
        Profielen met 'elevations', 'offsets', 'points' en 'crest_idx' keys.
        Moet eerst door detect_crest_points zijn verwerkt.
    smooth_sigma : float
        Sigma voor Gaussische smoothing voor knikpuntdetectie.

    Returns
    -------
    list[dict]
        Profielen aangevuld met per knikpunt-type:
        - '{type}_idx', '{type}_xy', '{type}_z'
    """
    for profile in profiles:
        z = profile["elevations"]
        pts = profile["points"]
        offsets = profile["offsets"]
        crest_idx = profile.get("crest_idx")

        # Init alle knikpunten als None
        for ktype in KNIKPUNT_TYPES:
            profile[f"{ktype}_idx"] = None
            profile[f"{ktype}_xy"] = None
            profile[f"{ktype}_z"] = None

        valid = np.isfinite(z)
        if valid.sum() < 10 or crest_idx is None:
            continue

        z_clean = _interpolate_nans(z.copy())
        z_smooth = gaussian_filter1d(z_clean, sigma=smooth_sigma)

        # Bereken 2e afgeleide (kromming)
        d2z = np.gradient(np.gradient(z_smooth))

        # Binnenzijde van de dijk (indices < crest_idx, negatieve offsets)
        # Buitenzijde van de dijk (indices > crest_idx, positieve offsets)
        margin = 3  # minimale afstand tot rand en kruin

        # --- Binnenzijde (links van kruin) ---
        inner = slice(margin, max(margin + 1, crest_idx - margin))
        d2z_inner = d2z[inner]

        if len(d2z_inner) > 2:
            # Binnenteen: sterkste positieve kromming (concaaf → convex overgang)
            # = waar het profiel begint te stijgen vanuit de polder
            pos_peaks, pos_props = find_peaks(d2z_inner, prominence=0.001)
            if len(pos_peaks) > 0:
                # Neem de meest prominente positieve krommingspiek
                best = pos_peaks[np.argmax(pos_props["prominences"])]
                idx = best + inner.start
                profile["binnenteen_idx"] = int(idx)
                profile["binnenteen_xy"] = tuple(pts[idx])
                profile["binnenteen_z"] = float(z[idx])

            # Kruinrand binnen: sterkste negatieve kromming (convex → concaaf)
            # = waar het talud overgaat in de kruin
            neg_peaks, neg_props = find_peaks(-d2z_inner, prominence=0.001)
            if len(neg_peaks) > 0:
                # Neem de piek het dichtst bij de kruin
                best = neg_peaks[-1]
                idx = best + inner.start
                profile["kruinrand_binnen_idx"] = int(idx)
                profile["kruinrand_binnen_xy"] = tuple(pts[idx])
                profile["kruinrand_binnen_z"] = float(z[idx])

        # --- Buitenzijde (rechts van kruin) ---
        outer_start = min(crest_idx + margin, len(z) - margin - 1)
        outer = slice(outer_start, len(z) - margin)
        d2z_outer = d2z[outer]

        if len(d2z_outer) > 2:
            # Kruinrand buiten: sterkste negatieve kromming
            # = waar de kruin overgaat in het buitentalud
            neg_peaks, neg_props = find_peaks(-d2z_outer, prominence=0.001)
            if len(neg_peaks) > 0:
                # Neem de piek het dichtst bij de kruin
                best = neg_peaks[0]
                idx = best + outer.start
                profile["kruinrand_buiten_idx"] = int(idx)
                profile["kruinrand_buiten_xy"] = tuple(pts[idx])
                profile["kruinrand_buiten_z"] = float(z[idx])

            # Buitenteen: sterkste positieve kromming
            # = waar het buitentalud overgaat in het voorland
            pos_peaks, pos_props = find_peaks(d2z_outer, prominence=0.001)
            if len(pos_peaks) > 0:
                best = pos_peaks[np.argmax(pos_props["prominences"])]
                idx = best + outer.start
                profile["buitenteen_idx"] = int(idx)
                profile["buitenteen_xy"] = tuple(pts[idx])
                profile["buitenteen_z"] = float(z[idx])

    return profiles


def knikpunten_to_lines(
    profiles: list[dict], smooth: bool = True
) -> dict[str, LineString | None]:
    """Verbind knikpunten tot kniklijnen per type.

    Returns
    -------
    dict[str, LineString | None]
        Dict met per knikpunt-type de resulterende lijn.
    """
    lines = {}
    for ktype in KNIKPUNT_TYPES:
        key = f"{ktype}_xy"
        points = [p[key] for p in profiles if p.get(key) is not None]
        if len(points) < 2:
            lines[ktype] = None
            continue

        coords = np.array(points)
        if smooth and len(coords) > 5:
            kernel = 5
            pad = kernel // 2
            coords_padded = np.pad(coords, ((pad, pad), (0, 0)), mode="edge")
            for i in range(len(coords)):
                coords[i] = coords_padded[i : i + kernel].mean(axis=0)

        lines[ktype] = LineString(coords)

    return lines


def crest_points_to_line(profiles: list[dict], smooth: bool = True) -> LineString | None:
    """Verbind gedetecteerde kruinpunten tot een kruinlijn.

    Parameters
    ----------
    profiles : list[dict]
        Profielen met 'crest_xy' key.
    smooth : bool
        Als True, pas een moving average toe om uitschieters te dempen.

    Returns
    -------
    LineString | None
        De kruinlijn, of None als er te weinig punten zijn.
    """
    points = [p["crest_xy"] for p in profiles if p.get("crest_xy") is not None]
    if len(points) < 2:
        return None

    coords = np.array(points)
    if smooth and len(coords) > 5:
        kernel = 5
        pad = kernel // 2
        coords_padded = np.pad(coords, ((pad, pad), (0, 0)), mode="edge")
        for i in range(len(coords)):
            coords[i] = coords_padded[i : i + kernel].mean(axis=0)

    return LineString(coords)


def _interpolate_nans(z: np.ndarray) -> np.ndarray:
    """Lineaire interpolatie van NaN-waarden."""
    nans = np.isnan(z)
    if nans.all():
        return z
    x = np.arange(len(z))
    z[nans] = np.interp(x[nans], x[~nans], z[~nans])
    return z


def _detect_peak(z: np.ndarray, min_prominence: float) -> int | None:
    """Vind de meest prominente piek (= kruinkandidaat)."""
    peaks, properties = find_peaks(z, prominence=min_prominence)
    if len(peaks) == 0:
        return None
    # Kies de piek met de hoogste prominentie
    best = peaks[np.argmax(properties["prominences"])]
    return int(best)


def _detect_curvature(z: np.ndarray, min_prominence: float) -> int | None:
    """Vind het kruinpunt via maximale negatieve kromming (2e afgeleide).

    De kruin is het punt waar de 2e afgeleide het meest negatief is
    (= maximale convexe kromming), mits het ook een lokale piek is.
    """
    # Tweede afgeleide (discrete benadering)
    d2z = np.gradient(np.gradient(z))

    # Zoek pieken in het hoogteprofiel
    peaks, properties = find_peaks(z, prominence=min_prominence * 0.5)
    if len(peaks) == 0:
        # Fallback: gewoon het hoogste punt
        return int(np.argmax(z))

    # Van de pieken, kies die met de meest negatieve kromming
    curvatures = d2z[peaks]
    best_peak = peaks[np.argmin(curvatures)]
    return int(best_peak)
