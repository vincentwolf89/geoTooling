"""Detecteer kruinpunten en knikpunten op dwarsprofielen via morfologische analyse."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from shapely.geometry import LineString, Point

# Knikpunt-types die gedetecteerd worden
KNIKPUNT_TYPES = [
    "binnenteen",   # overgang polder → binnentalud
    "binnenberm",   # vlak stuk op binnentalud (optioneel, kan None zijn)
    "binnenkruin",  # overgang binnentalud → kruin
    "buitenkruin",  # overgang kruin → buitentalud
    "buitenberm",   # vlak stuk op buitentalud (optioneel, kan None zijn)
    "buitenteen",   # overgang buitentalud → voorland/water
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
    water_side: str | None = None,
) -> list[dict]:
    """Detecteer alle knikpunten (breeklijnen) per dwarsprofiel.

    Knikpunten zijn locaties waar de helling significant verandert:
    - binnenteen: overgang polder → binnentalud (positieve krommingspiek, landzijde)
    - binnenkruin: overgang binnentalud → kruin (negatieve krommingspiek, landzijde)
    - buitenkruin: overgang kruin → buitentalud (negatieve krommingspiek, waterzijde)
    - buitenteen: overgang buitentalud → voorland/water (positieve krommingspiek, waterzijde)

    Binnen/buiten kan expliciet opgegeven worden via ``water_side``, of wordt
    automatisch bepaald: de zijde met de laagste randhoogte = binnenzijde (polder).

    Parameters
    ----------
    profiles : list[dict]
        Profielen met 'elevations', 'offsets', 'points' en 'crest_idx' keys.
        Moet eerst door detect_crest_points zijn verwerkt.
    smooth_sigma : float
        Sigma voor Gaussische smoothing voor knikpuntdetectie.
    water_side : str | None
        Welke zijde van het profiel de waterzijde (buitenzijde) is:
        - 'left' of 'right': forceer de buitenzijde
        - None: automatische detectie op basis van randhoogte

    Returns
    -------
    list[dict]
        Profielen aangevuld met per knikpunt-type:
        - '{type}_idx', '{type}_xy', '{type}_z'
    """
    # Stap 1: als water_side niet opgegeven, bepaal het GLOBAAL
    # via meerderheidsstemming over alle profielen. Dit voorkomt dat
    # de binnen/buiten-toewijzing per profiel wisselt (= kruisende lijnen).
    if water_side is None:
        votes_left_is_binnen = 0
        votes_right_is_binnen = 0
        margin = 3
        edge_n = min(5, margin + 1)

        for profile in profiles:
            z = profile["elevations"]
            crest_idx = profile.get("crest_idx")
            if crest_idx is None or np.isfinite(z).sum() < 10:
                continue
            z_clean = _interpolate_nans(z.copy())
            z_smooth = gaussian_filter1d(z_clean, sigma=smooth_sigma)
            left_edge_z = np.nanmean(z_smooth[:edge_n])
            right_edge_z = np.nanmean(z_smooth[-edge_n:])
            if left_edge_z <= right_edge_z:
                votes_left_is_binnen += 1
            else:
                votes_right_is_binnen += 1

        global_left_is_binnen = votes_left_is_binnen >= votes_right_is_binnen

    # Stap 2: detecteer knikpunten per profiel met consistente zijde-toewijzing
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

        margin = 3  # minimale afstand tot rasterrand

        # Detecteer knikpunten op beide zijden van de kruin
        # Met max afstand beperking zodat we niet buiten de dijk zoeken
        left_kruinrand, left_teen = _detect_side_knikpunten(
            z_smooth, d2z, margin, crest_idx,
            is_left=True,
            offsets=offsets, crest_idx=crest_idx,
        )
        right_kruinrand, right_teen = _detect_side_knikpunten(
            z_smooth, d2z, crest_idx + 1, len(z) - margin,
            is_left=False,
            offsets=offsets, crest_idx=crest_idx,
        )

        # Detecteer bermen (vlakke stukken op het talud)
        d1z = np.gradient(z_smooth)
        left_berm = _detect_berm(d1z, left_teen, left_kruinrand)
        right_berm = _detect_berm(d1z, right_teen, right_kruinrand)

        # Bepaal welke zijde binnen (polder) en buiten (water) is
        # Gebruik de GLOBALE bepaling zodat alle profielen consistent zijn
        if water_side == "right":
            left_is_binnen = True
        elif water_side == "left":
            left_is_binnen = False
        else:
            left_is_binnen = global_left_is_binnen

        if left_is_binnen:
            binnen_kruinrand, binnen_teen, binnen_berm = left_kruinrand, left_teen, left_berm
            buiten_kruinrand, buiten_teen, buiten_berm = right_kruinrand, right_teen, right_berm
        else:
            binnen_kruinrand, binnen_teen, binnen_berm = right_kruinrand, right_teen, right_berm
            buiten_kruinrand, buiten_teen, buiten_berm = left_kruinrand, left_teen, left_berm

        _assign_knikpunt(profile, "binnenteen", binnen_teen, pts, z)
        _assign_knikpunt(profile, "binnenberm", binnen_berm, pts, z)
        _assign_knikpunt(profile, "binnenkruin", binnen_kruinrand, pts, z)
        _assign_knikpunt(profile, "buitenkruin", buiten_kruinrand, pts, z)
        _assign_knikpunt(profile, "buitenberm", buiten_berm, pts, z)
        _assign_knikpunt(profile, "buitenteen", buiten_teen, pts, z)

    return profiles


def _detect_side_knikpunten(
    z_smooth: np.ndarray,
    d2z: np.ndarray,
    start: int,
    end: int,
    is_left: bool,
    offsets: np.ndarray | None = None,
    crest_idx: int | None = None,
    max_kruinrand_dist: float = 8.0,
    max_teen_dist: float = 25.0,
) -> tuple[int | None, int | None]:
    """Detecteer kruinrand en teen op één zijde van de kruin.

    Parameters
    ----------
    max_kruinrand_dist : float
        Maximale afstand (m) van de kruinrand tot de kruin.
    max_teen_dist : float
        Maximale afstand (m) van de teen tot de kruin.
        Voorkomt dat de detectie buiten de dijk gaat.

    Returns
    -------
    (kruinrand_idx, teen_idx) : tuple[int | None, int | None]
        Globale indices, of None als niet gedetecteerd.
    """
    if end <= start or end - start < 3:
        return None, None

    side_d2z = d2z[start:end]

    kruinrand_idx = None
    teen_idx = None

    # Bepaal afstandslimiet in indices (als offsets beschikbaar)
    crest_offset = offsets[crest_idx] if offsets is not None and crest_idx is not None else None

    def _within_max_dist(idx: int, max_dist: float) -> bool:
        """Check of index binnen maximale afstand van kruin ligt."""
        if crest_offset is None or offsets is None:
            return True
        return abs(offsets[idx] - crest_offset) <= max_dist

    # Kruinrand: negatieve krommingspiek dichtst bij de kruin
    neg_peaks, neg_props = find_peaks(-side_d2z, prominence=0.002)
    if len(neg_peaks) > 0:
        global_neg = neg_peaks + start
        # Filter op maximale afstand
        valid = [i for i, gp in enumerate(global_neg)
                 if _within_max_dist(gp, max_kruinrand_dist)]
        if valid:
            # Links: dichtst bij kruin = hoogste index; rechts: laagste
            best = valid[-1] if is_left else valid[0]
            kruinrand_idx = int(global_neg[best])

    # Teen: positieve krommingspiek, verder van kruin dan kruinrand
    pos_peaks, pos_props = find_peaks(side_d2z, prominence=0.002)
    if len(pos_peaks) > 0:
        global_peaks = pos_peaks + start
        prominences = pos_props["prominences"]

        # Filter 1: teen moet verder van de kruin liggen dan kruinrand
        if kruinrand_idx is not None:
            if is_left:
                mask = global_peaks < kruinrand_idx
            else:
                mask = global_peaks > kruinrand_idx
        else:
            mask = np.ones(len(pos_peaks), dtype=bool)

        # Filter 2: teen moet binnen maximale afstand van kruin
        dist_mask = np.array([_within_max_dist(gp, max_teen_dist)
                              for gp in global_peaks])
        mask = mask & dist_mask

        if mask.any():
            # Neem de meest prominente kandidaat
            best_i = np.argmax(prominences[mask])
            teen_idx = int(global_peaks[mask][best_i])

    return kruinrand_idx, teen_idx


def _detect_berm(
    d1z: np.ndarray,
    teen_idx: int | None,
    kruinrand_idx: int | None,
    min_width: int = 3,
    slope_ratio: float = 0.3,
) -> int | None:
    """Detecteer een berm (vlak stuk) op het talud tussen teen en kruinrand.

    Een berm is een zone waar de helling significant lager is dan de
    gemiddelde helling op het talud. Geeft None als er geen berm is.

    Parameters
    ----------
    d1z : np.ndarray
        Eerste afgeleide (helling) van het gesmoothe profiel.
    teen_idx, kruinrand_idx : int | None
        Globale indices van teen en kruinrand. Als een van beide None is,
        kan geen berm gedetecteerd worden.
    min_width : int
        Minimaal aantal samples dat de vlakke zone breed moet zijn.
    slope_ratio : float
        De minimale helling in de bermzone moet lager zijn dan dit
        aandeel van de gemiddelde helling op het talud.
    """
    if teen_idx is None or kruinrand_idx is None:
        return None

    lo, hi = sorted([teen_idx, kruinrand_idx])
    if hi - lo < min_width + 2:
        return None

    # Absolute helling op het talud (exclusief de randen)
    talud_slope = np.abs(d1z[lo + 1 : hi])
    if len(talud_slope) < min_width:
        return None

    mean_slope = talud_slope.mean()
    if mean_slope < 1e-6:
        return None

    # Zoek de positie met minimale helling
    min_idx_local = np.argmin(talud_slope)
    min_slope = talud_slope[min_idx_local]

    # Check of de minimale helling duidelijk lager is dan gemiddeld
    if min_slope > slope_ratio * mean_slope:
        return None

    # Check dat er voldoende breedte is rond het minimum
    threshold = slope_ratio * mean_slope
    flat_mask = talud_slope < threshold
    # Zoek de aaneengesloten regio rond het minimum
    run_start = min_idx_local
    while run_start > 0 and flat_mask[run_start - 1]:
        run_start -= 1
    run_end = min_idx_local
    while run_end < len(flat_mask) - 1 and flat_mask[run_end + 1]:
        run_end += 1

    if run_end - run_start + 1 < min_width:
        return None

    # Berm-punt = midden van de vlakke zone
    berm_local = (run_start + run_end) // 2
    return int(berm_local + lo + 1)


def _assign_knikpunt(
    profile: dict, ktype: str, idx: int | None,
    pts: np.ndarray, z: np.ndarray,
) -> None:
    """Wijs een knikpunt toe aan het profiel."""
    if idx is not None:
        profile[f"{ktype}_idx"] = idx
        profile[f"{ktype}_xy"] = tuple(pts[idx])
        profile[f"{ktype}_z"] = float(z[idx])


def _filter_outliers_and_smooth(coords: np.ndarray, max_deviation: float = 8.0) -> np.ndarray:
    """Verwijder uitschieters en smooth coördinaten.

    Uitschieters worden gedetecteerd als punten die meer dan max_deviation
    meter afwijken van het lopende mediaan. Na filtering wordt een Gaussian
    smooth toegepast voor een vloeiend resultaat.

    Parameters
    ----------
    coords : np.ndarray
        (N, 2) array van XY-coördinaten.
    max_deviation : float
        Maximale afwijking in meters t.o.v. het lokale mediaan.
    """
    if len(coords) < 5:
        return coords

    # Stap 1: median filter om referentielijn te bepalen
    from scipy.ndimage import median_filter, gaussian_filter1d

    window = min(11, len(coords) // 2 * 2 + 1)  # oneven window
    ref_x = median_filter(coords[:, 0], size=window, mode="nearest")
    ref_y = median_filter(coords[:, 1], size=window, mode="nearest")

    # Stap 2: afwijking t.o.v. mediaan
    dx = coords[:, 0] - ref_x
    dy = coords[:, 1] - ref_y
    dist = np.sqrt(dx**2 + dy**2)

    # Stap 3: verwijder uitschieters
    inliers = dist < max_deviation
    if inliers.sum() < 2:
        return coords

    # Interpoleer gaps waar uitschieters verwijderd zijn
    x_clean = np.interp(
        np.arange(len(coords)),
        np.where(inliers)[0],
        coords[inliers, 0],
    )
    y_clean = np.interp(
        np.arange(len(coords)),
        np.where(inliers)[0],
        coords[inliers, 1],
    )

    # Stap 4: Gaussian smoothing
    sigma = max(3.0, len(coords) / 50)
    x_smooth = gaussian_filter1d(x_clean, sigma=sigma)
    y_smooth = gaussian_filter1d(y_clean, sigma=sigma)

    return np.column_stack([x_smooth, y_smooth])


def knikpunten_to_lines(
    profiles: list[dict], smooth: bool = True
) -> dict[str, LineString | None]:
    """Verbind knikpunten tot kniklijnen per type.

    Filtert uitschieters en past Gaussian smoothing toe om te voorkomen
    dat lijnen elkaar kruisen.

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
            coords = _filter_outliers_and_smooth(coords)

        lines[ktype] = LineString(coords)

    return lines


def crest_points_to_line(profiles: list[dict], smooth: bool = True) -> LineString | None:
    """Verbind gedetecteerde kruinpunten tot een kruinlijn.

    Parameters
    ----------
    profiles : list[dict]
        Profielen met 'crest_xy' key.
    smooth : bool
        Als True, filter uitschieters en pas Gaussian smoothing toe.

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
        coords = _filter_outliers_and_smooth(coords)

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
