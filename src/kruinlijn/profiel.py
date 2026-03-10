"""Knikpunt-detectie via dwarsprofiel-analyse op AHN DTM.

Extraheert binnenkruinlijn, buitenkruinlijn, binnenteenlijn, buitenteenlijn
door elke meter een dwarsprofiel te samplen en knikpunten te vinden via
krommingsanalyse (2e afgeleide).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import LineString, Point


def _perpendicular_profile(trajectory, distance, half_width=120.0, step=0.5):
    """Sample een dwarsprofiel loodrecht op het traject."""
    pt = trajectory.interpolate(distance)
    d_fwd = min(distance + 5, trajectory.length)
    d_bwd = max(distance - 5, 0)
    p_fwd = trajectory.interpolate(d_fwd)
    p_bwd = trajectory.interpolate(d_bwd)

    dx = p_fwd.x - p_bwd.x
    dy = p_fwd.y - p_bwd.y
    length = (dx**2 + dy**2) ** 0.5
    if length < 0.01:
        return None, None, None

    nx = -dy / length
    ny = dx / length

    offsets = np.arange(-half_width, half_width + step, step)
    xs = pt.x + offsets * nx
    ys = pt.y + offsets * ny
    return xs, ys, offsets


def _sample_dtm(xs, ys, dtm, transform):
    """Sample DTM-waarden op gegeven coordinaten."""
    inv = ~transform
    cols, rows = inv * (xs, ys)
    cols = np.round(cols).astype(int)
    rows = np.round(rows).astype(int)

    h, w = dtm.shape
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)

    elevations = np.full(len(xs), np.nan)
    elevations[valid] = dtm[rows[valid], cols[valid]]
    elevations[(elevations < -10) | (elevations > 100)] = np.nan
    return elevations


def _determine_buiten_side(trajectory, dtm, transform, n_samples=50):
    """Bepaal eenmalig welke kant 'buiten' (rivierzijde) is.

    Gebruikt twee heuristieken:
    1. Taludsteilheid: buitentalud is steiler (1:3) dan binnentalud (1:2)
    2. Hoogte ver weg: buitenzijde is doorgaans lager
    Combineert beide via gewogen stemming.
    """
    total = trajectory.length
    votes_left = 0
    votes_right = 0

    for i in range(n_samples):
        dist = total * (i + 1) / (n_samples + 1)
        xs, ys, offsets = _perpendicular_profile(trajectory, dist, half_width=120.0)
        if xs is None:
            continue

        elevations = _sample_dtm(xs, ys, dtm, transform)
        valid = ~np.isnan(elevations)

        center = np.abs(offsets) < 40
        if not (center & valid).any():
            continue

        elev_c = elevations.copy()
        elev_c[~valid | ~center] = -999
        kruin_idx = np.argmax(elev_c)
        kruin_off = offsets[kruin_idx]
        kruin_elev = elevations[kruin_idx]

        elev_smooth = gaussian_filter1d(
            np.where(valid, elevations, np.nanmedian(elevations)),
            sigma=3.0,
        )
        slope = np.abs(np.gradient(elev_smooth, offsets))

        # Talud zones: 5-25m van kruin op elke kant
        left_talud = (offsets < kruin_off - 5) & (offsets > kruin_off - 25) & valid
        right_talud = (offsets > kruin_off + 5) & (offsets < kruin_off + 25) & valid

        if left_talud.sum() > 3 and right_talud.sum() > 3:
            left_steepness = np.mean(slope[left_talud])
            right_steepness = np.mean(slope[right_talud])
            # Steilere kant = buiten (zwaarder gewogen)
            if left_steepness > right_steepness * 1.1:
                votes_left += 2
            elif right_steepness > left_steepness * 1.1:
                votes_right += 2

        # Hoogte ver weg: 60-100m
        far_left = (offsets < kruin_off - 60) & (offsets > kruin_off - 100) & valid
        far_right = (offsets > kruin_off + 60) & (offsets < kruin_off + 100) & valid
        if far_left.sum() > 3 and far_right.sum() > 3:
            if np.nanmedian(elevations[far_left]) < np.nanmedian(elevations[far_right]):
                votes_left += 1
            else:
                votes_right += 1

    result = "left" if votes_left >= votes_right else "right"
    print(f"  Buiten-kant: {'links' if result == 'left' else 'rechts'} "
          f"(L={votes_left}, R={votes_right})")
    return result


def _profile_quality(offsets, elevations, sigma=3.0):
    """Geef een kwaliteitsscore (0-1) voor een dwarsprofiel.

    Slechte profielen (bebouwing, opritten, dorpskernen) worden afgewezen.
    Criteria:
    - Voldoende geldige waarden (>60%)
    - Duidelijke kruin (uitstekend boven omgeving)
    - Beperkte hoogtevariatie (geen gebouwen = plotse sprongen)
    - Vloeiend profiel (lage ruwheid)
    """
    valid = ~np.isnan(elevations)
    valid_frac = valid.sum() / len(elevations)
    if valid_frac < 0.5:
        return 0.0

    center = np.abs(offsets) < 35
    if not (center & valid).any():
        return 0.0

    elev = elevations.copy()
    elev[~valid] = np.nanmedian(elevations)
    elev_smooth = gaussian_filter1d(elev, sigma=sigma)

    # Kruin moet duidelijk hoger zijn dan flanken
    elev_center = elev_smooth.copy()
    elev_center[~center] = -999
    kruin_idx = np.argmax(elev_center)
    kruin_elev = elev_smooth[kruin_idx]

    # Hoogte links en rechts (30-80m van kruin)
    kruin_off = offsets[kruin_idx]
    far_left = (offsets < kruin_off - 30) & (offsets > kruin_off - 80) & valid
    far_right = (offsets > kruin_off + 30) & (offsets < kruin_off + 80) & valid
    if far_left.sum() < 5 or far_right.sum() < 5:
        return 0.3

    left_elev = np.nanmedian(elevations[far_left])
    right_elev = np.nanmedian(elevations[far_right])
    prominence = kruin_elev - min(left_elev, right_elev)

    if prominence < 1.0:
        return 0.1  # Geen duidelijke dijk

    # Ruwheid: plotse hoogtesprongen wijzen op gebouwen
    diffs = np.abs(np.diff(elev_smooth))
    # Grote sprongen (>1m per 0.5m stap = onnatuurlijk)
    n_jumps = (diffs > 1.0).sum()
    if n_jumps > 10:
        return 0.2

    # Hoogtebereik in centraal deel (gebouwen geven extreem bereik)
    center_elev = elevations[center & valid]
    if len(center_elev) > 10:
        center_range = np.percentile(center_elev, 95) - np.percentile(center_elev, 5)
        if center_range > 8.0:  # Meer dan 8m variatie centraal = gebouwen
            return 0.2

    score = min(1.0, prominence / 3.0) * min(1.0, valid_frac / 0.7)
    return score


def _find_knikpunten(offsets, elevations, buiten_side, sigma=3.0):
    """Vind knikpunten via kromming (kruinranden) + helling-wandeling (teenlijnen).

    Kruinranden: sterkste negatieve kromming nabij kruintop (convex knik).
    Teenlijnen: wandel talud af, zoek waar helling afvlakt EN hoogte < kruin - 1.5m.
    """
    valid = ~np.isnan(elevations)
    if valid.sum() < 30:
        return None

    # Kwaliteitscheck: wijs slechte profielen af
    quality = _profile_quality(offsets, elevations, sigma)
    if quality < 0.4:
        return None

    step = offsets[1] - offsets[0]  # typisch 0.5m

    # Smooth profiel
    elev = elevations.copy()
    elev[~valid] = np.nanmedian(elevations)
    elev_smooth = gaussian_filter1d(elev, sigma=sigma)

    # Helling en kromming
    slope = np.gradient(elev_smooth, offsets)
    abs_slope = gaussian_filter1d(np.abs(slope), sigma=2.0)
    curvature = np.gradient(slope, offsets)
    curv_smooth = gaussian_filter1d(curvature, sigma=sigma)

    # Kruin: hoogste punt centraal
    center = np.abs(offsets) < 35
    if not (center & valid).any():
        return None

    elev_center = elev_smooth.copy()
    elev_center[~center] = -999
    kruin_idx = np.argmax(elev_center)
    kruin_offset = offsets[kruin_idx]
    kruin_elev = elev_smooth[kruin_idx]

    # Buiten/binnen zones vanuit kruin
    if buiten_side == "left":
        buiten_mask = offsets < kruin_offset - 1
        binnen_mask = offsets > kruin_offset + 1
    else:
        buiten_mask = offsets > kruin_offset + 1
        binnen_mask = offsets < kruin_offset - 1

    # --- Kruinranden via kromming ---
    # Kruinrand = sterkste negatieve kromming nabij kruin (convex knik)
    def find_kruin_edge(side_mask, max_dist=15.0):
        near_kruin = side_mask & (np.abs(offsets - kruin_offset) < max_dist)
        if near_kruin.sum() < 3:
            return None

        neg_curv = -curv_smooth.copy()
        neg_curv[~near_kruin] = -999

        idx = np.argmax(neg_curv)
        if neg_curv[idx] <= 0:
            # Fallback: punt waar helling > 0.05 begint
            for i in np.where(near_kruin)[0]:
                if abs_slope[i] > 0.05:
                    return i
            return None
        return idx

    buitenkruin_idx = find_kruin_edge(buiten_mask)
    binnenkruin_idx = find_kruin_edge(binnen_mask)

    if buitenkruin_idx is None or binnenkruin_idx is None:
        return None

    # Kruinranden moeten correct geordend zijn (buiten verder weg dan binnen)
    if buiten_side == "left":
        if offsets[buitenkruin_idx] >= offsets[binnenkruin_idx]:
            return None
    else:
        if offsets[buitenkruin_idx] <= offsets[binnenkruin_idx]:
            return None

    # Minimale kruinbreedte: 2m
    if abs(offsets[binnenkruin_idx] - offsets[buitenkruin_idx]) < 2.0:
        return None

    # --- Teenlijnen via helling-wandeling ---
    # Wandel het talud af. Teen = LAATSTE vlakke plek na steil talud (passeer bermen).
    SLOPE_STEEP = 0.05
    SLOPE_FLAT = 0.05

    def walk_to_teen(kruin_edge_idx, direction, max_dist=60.0, find_lowest=False):
        """Wandel talud af, zoek teen.

        find_lowest=False: stop bij eerste vlakke plek (buitenzijde)
        find_lowest=True: zoek laagste vlakke plek, passeer bermen (binnenzijde)
        """
        edge_offset = offsets[kruin_edge_idx]
        steps = int(max_dist / abs(step))
        idx = kruin_edge_idx
        passed_steep = False
        best_teen = None
        best_teen_elev = kruin_elev

        for _ in range(steps):
            idx += direction
            if idx < 0 or idx >= len(offsets):
                break
            if not valid[idx]:
                continue

            dist = abs(offsets[idx] - edge_offset)

            # Moet eerst door steil stuk (talud)
            if abs_slope[idx] > SLOPE_STEEP and dist > 2:
                passed_steep = True

            # Kandidaat teen: na steil stuk, helling vlak, hoogte laag
            if (passed_steep
                    and abs_slope[idx] < SLOPE_FLAT
                    and elev_smooth[idx] < kruin_elev - 1.5):
                if not find_lowest:
                    # Buitenzijde: eerste vlakke plek = teen
                    return idx

                # Binnenzijde: bewaar de laagste vlakke plek
                if elev_smooth[idx] < best_teen_elev:
                    best_teen = idx
                    best_teen_elev = elev_smooth[idx]
                # Stop als hoogte weer stijgt (>0.5m boven minimum)
                if best_teen is not None and elev_smooth[idx] > best_teen_elev + 0.5:
                    break

        if best_teen is not None:
            return best_teen

        # Fallback: laagste punt in zone met hoogte < kruin - 1.5m
        idx = kruin_edge_idx
        for _ in range(steps):
            idx += direction
            if idx < 0 or idx >= len(offsets):
                break
            dist = abs(offsets[idx] - edge_offset)
            if dist < 5 or not valid[idx]:
                continue
            if elev_smooth[idx] < kruin_elev - 1.5 and elev_smooth[idx] < best_teen_elev:
                best_teen_elev = elev_smooth[idx]
                best_teen = idx
        return best_teen

    if buiten_side == "left":
        buitenteen_idx = walk_to_teen(buitenkruin_idx, -1, find_lowest=False)
        binnenteen_idx = walk_to_teen(binnenkruin_idx, +1, find_lowest=True)
    else:
        buitenteen_idx = walk_to_teen(buitenkruin_idx, +1, find_lowest=False)
        binnenteen_idx = walk_to_teen(binnenkruin_idx, -1, find_lowest=True)

    if buitenteen_idx is None or binnenteen_idx is None:
        return None

    # --- Bermen detectie ---
    # Berm = vlak stuk op het talud tussen kruinrand en teen.
    # Detectie: zoek eerste vlakke plek na steil talud, maar VOOR de teen.
    def find_berm(kruin_edge_idx, teen_idx, direction):
        """Zoek berm tussen kruinrand en teen."""
        edge_offset = offsets[kruin_edge_idx]
        teen_offset = offsets[teen_idx]
        idx = kruin_edge_idx
        passed_steep = False

        while True:
            idx += direction
            if idx < 0 or idx >= len(offsets):
                return None
            # Stop voor de teen
            if (direction > 0 and offsets[idx] >= teen_offset - 2) or \
               (direction < 0 and offsets[idx] <= teen_offset + 2):
                return None
            if not valid[idx]:
                continue

            dist = abs(offsets[idx] - edge_offset)

            if abs_slope[idx] > SLOPE_STEEP and dist > 2:
                passed_steep = True

            # Berm: vlak stuk na steil talud, minimaal 1m lager dan kruin
            if (passed_steep
                    and abs_slope[idx] < SLOPE_FLAT
                    and elev_smooth[idx] < kruin_elev - 1.0
                    and dist > 3):
                return idx
        return None

    binnenberm_idx = None
    buitenberm_idx = None
    if buiten_side == "left":
        binnenberm_idx = find_berm(binnenkruin_idx, binnenteen_idx, +1)
        buitenberm_idx = find_berm(buitenkruin_idx, buitenteen_idx, -1)
    else:
        binnenberm_idx = find_berm(binnenkruin_idx, binnenteen_idx, -1)
        buitenberm_idx = find_berm(buitenkruin_idx, buitenteen_idx, +1)

    # --- Validatie ---
    bk_o = offsets[buitenkruin_idx]
    ik_o = offsets[binnenkruin_idx]
    bt_o = offsets[buitenteen_idx]
    it_o = offsets[binnenteen_idx]

    # Hoogteverschil kruin-teen moet minimaal 1.5m zijn
    if kruin_elev - elev_smooth[buitenteen_idx] < 1.5:
        return None
    if kruin_elev - elev_smooth[binnenteen_idx] < 1.5:
        return None

    # Volgorde check
    if buiten_side == "left":
        if not (bt_o < bk_o < ik_o < it_o):
            return None
    else:
        if not (it_o < ik_o < bk_o < bt_o):
            return None

    # Minimale afstand kruin-teen: 5m
    if abs(bt_o - bk_o) < 5 or abs(it_o - ik_o) < 5:
        return None

    result = {
        "buitenkruin": bk_o,
        "binnenkruin": ik_o,
        "buitenteen": bt_o,
        "binnenteen": it_o,
    }

    # Bermen alleen toevoegen als ze echt tussen kruin en teen liggen
    if binnenberm_idx is not None:
        bb_o = offsets[binnenberm_idx]
        # Check dat berm echt tussen kruinrand en teen zit
        if buiten_side == "left":
            if ik_o < bb_o < it_o and abs(bb_o - ik_o) > 3 and abs(it_o - bb_o) > 3:
                result["binnenberm"] = bb_o
        else:
            if it_o < bb_o < ik_o and abs(bb_o - ik_o) > 3 and abs(it_o - bb_o) > 3:
                result["binnenberm"] = bb_o

    if buitenberm_idx is not None:
        ub_o = offsets[buitenberm_idx]
        if buiten_side == "left":
            if bt_o < ub_o < bk_o and abs(ub_o - bk_o) > 3 and abs(bt_o - ub_o) > 3:
                result["buitenberm"] = ub_o
        else:
            if bk_o < ub_o < bt_o and abs(ub_o - bk_o) > 3 and abs(bt_o - ub_o) > 3:
                result["buitenberm"] = ub_o

    return result



def _snap_to_terrain(x, y, dtm, transform, feature_type, search_radius=3.0):
    """Snap een knikpunt naar de beste terreinpositie op het AHN.

    feature_type:
      'kruinrand' -> zoek sterkste negatieve kromming (convexe knik)
      'teen' -> zoek vlakste plek (laagste helling)
    """
    step = abs(transform.a)  # pixelgrootte (~0.5m)
    n_pix = int(search_radius / step) + 1

    inv = ~transform
    col_c, row_c = inv * (x, y)
    col_c, row_c = int(round(col_c)), int(round(row_c))

    h, w = dtm.shape
    r0 = max(0, row_c - n_pix)
    r1 = min(h, row_c + n_pix + 1)
    c0 = max(0, col_c - n_pix)
    c1 = min(w, col_c + n_pix + 1)

    if r1 <= r0 or c1 <= c0:
        return x, y

    patch = dtm[r0:r1, c0:c1].astype(float)
    valid = (patch > -10) & (patch < 100)
    if valid.sum() < 5:
        return x, y

    patch[~valid] = np.nan

    if feature_type == "kruinrand":
        # Zoek sterkste kromming (2e afgeleide, negatief = convex)
        dy_g, dx_g = np.gradient(patch, step)
        dyy, _ = np.gradient(dy_g, step)
        _, dxx = np.gradient(dx_g, step)
        curv = -(dxx + dyy)  # negatief = convex knik
        curv[~valid] = -999
        best = np.unravel_index(np.argmax(curv), curv.shape)
    else:
        # Teen: zoek vlakste plek (laagste gradient magnitude)
        dy_g, dx_g = np.gradient(patch, step)
        slope_mag = np.sqrt(dx_g**2 + dy_g**2)
        slope_mag[~valid] = 999
        best = np.unravel_index(np.argmin(slope_mag), slope_mag.shape)

    # Terug naar wereld-coordinaten
    best_row = r0 + best[0]
    best_col = c0 + best[1]
    snap_x, snap_y = transform * (best_col, best_row)
    return snap_x, snap_y


def extract_kniklijnen(
    trajectory: LineString,
    dtm_path: str | Path,
    profile_spacing: float = 1.0,
    half_width: float = 120.0,
    smooth_sigma: float = 3.0,
    line_smooth_sigma: float = 10.0,
) -> tuple[dict[str, LineString], dict[str, list], any]:
    """Extraheer kniklijnen via dwarsprofiel-analyse.

    Returns:
        (lines_dict, points_dict, crs)
        lines_dict: {naam: LineString} voor de 4 kniklijnen
        points_dict: {naam: [(x,y,z), ...]} raw knikpunten
    """
    with rasterio.open(dtm_path) as src:
        dtm = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs

    if trajectory.has_z:
        trajectory = LineString([(x, y) for x, y, *_ in trajectory.coords])

    total_length = trajectory.length
    distances = np.arange(0, total_length, profile_spacing)

    buiten_side = _determine_buiten_side(trajectory, dtm, transform)

    # Stap 1: Verzamel raw knikpunten als XY coordinaten
    all_names = ["binnenkruin", "buitenkruin", "binnenteen", "buitenteen"]
    raw_points = {name: [] for name in all_names}  # [(x, y, dist_along)]
    raw_dists = {name: [] for name in all_names}

    n_found = 0
    for dist in distances:
        xs, ys, offsets = _perpendicular_profile(trajectory, dist, half_width=half_width)
        if xs is None:
            continue

        elevations = _sample_dtm(xs, ys, dtm, transform)
        knik = _find_knikpunten(offsets, elevations, buiten_side, sigma=smooth_sigma)

        if knik is None:
            continue

        n_found += 1

        # Converteer offset naar XY
        pt = trajectory.interpolate(dist)
        d_fwd = min(dist + 5, total_length)
        d_bwd = max(dist - 5, 0)
        p_fwd = trajectory.interpolate(d_fwd)
        p_bwd = trajectory.interpolate(d_bwd)
        dx = p_fwd.x - p_bwd.x
        dy = p_fwd.y - p_bwd.y
        length = (dx**2 + dy**2) ** 0.5
        if length < 0.01:
            continue
        nx = -dy / length
        ny = dx / length

        for name in all_names:
            off = knik[name]
            x = pt.x + off * nx
            y = pt.y + off * ny
            raw_points[name].append((x, y))
            raw_dists[name].append(dist)

    print(f"  {n_found}/{len(distances)} profielen ({n_found/len(distances):.0%})")

    if n_found < 10:
        return {n + "lijn": None for n in all_names}, {n: [] for n in all_names}, crs

    # Stap 2: Snap punten naar AHN terreinkenmerken
    snapped_points = {name: [] for name in all_names}
    for name in all_names:
        feat_type = "kruinrand" if "kruin" in name else "teen"
        for x, y in raw_points[name]:
            sx, sy = _snap_to_terrain(x, y, dtm, transform, feat_type, search_radius=3.0)
            snapped_points[name].append((sx, sy))

    print(f"  Punten gesnapt naar AHN")

    # Stap 3: Bouw gladde lijnen door gesnappte punten
    # Smooth X en Y apart met gaussian langs traject-parameter
    lines = {}
    points_out = {}
    for name in all_names:
        pts = np.array(snapped_points[name])
        dists = np.array(raw_dists[name])

        if len(pts) < 10:
            lines[name + "lijn"] = None
            points_out[name] = []
            continue

        # Bewaar raw punten voor output (met hoogte)
        pts_with_z = []
        inv = ~transform
        for px, py in pts:
            col, row = inv * (px, py)
            col, row = int(round(col)), int(round(row))
            h, w = dtm.shape
            if 0 <= row < h and 0 <= col < w:
                z = float(dtm[row, col])
            else:
                z = 0.0
            pts_with_z.append((px, py, z))
        points_out[name] = pts_with_z

        # Outlier filter op XY: verwijder punten die ver van smooth referentie liggen
        x_smooth = gaussian_filter1d(pts[:, 0], sigma=15, mode="nearest")
        y_smooth = gaussian_filter1d(pts[:, 1], sigma=15, mode="nearest")
        dist_from_smooth = np.sqrt((pts[:, 0] - x_smooth)**2 + (pts[:, 1] - y_smooth)**2)
        threshold = max(np.percentile(dist_from_smooth, 85) * 3.0, 5.0)
        good = dist_from_smooth < threshold

        if good.sum() < 10:
            lines[name + "lijn"] = None
            continue

        pts_good = pts[good]

        # Gaussian smooth op X en Y apart
        # Kruinlijnen: meer smoothing (moeten vloeiend zijn, smalle kruin)
        # Teenlijnen: minder smoothing (moeten terrein volgen)
        if "kruin" in name:
            sigma = min(line_smooth_sigma * 2, len(pts_good) / 5)
        else:
            sigma = min(line_smooth_sigma, len(pts_good) / 5)
        x_final = gaussian_filter1d(pts_good[:, 0], sigma=sigma, mode="nearest")
        y_final = gaussian_filter1d(pts_good[:, 1], sigma=sigma, mode="nearest")

        coords = list(zip(x_final, y_final))
        line = LineString(coords)
        line = line.simplify(2.0, preserve_topology=True)
        lines[name + "lijn"] = line
        print(f"  {name}lijn: {line.length:.0f}m ({good.sum()} punten)")

    # Voeg ontbrekende lijnen toe als None
    for name in all_names:
        if name + "lijn" not in lines:
            lines[name + "lijn"] = None

    return lines, points_out, crs


