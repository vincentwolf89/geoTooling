"""Detecteer kruinpunten op dwarsprofielen via morfologische analyse."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from shapely.geometry import LineString, Point


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
