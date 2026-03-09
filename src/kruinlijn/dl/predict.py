"""Voorspelling met een getraind segmentatiemodel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch

from .dataset import (
    _normalize, _compute_slope, _compute_aspect,
    _compute_curvature, _compute_tpi, NUM_CLASSES,
)
from .model import build_unet


def predict_tiles(
    dtm_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    rgb_path: str | Path | None = None,
    dsm_path: str | Path | None = None,
    tile_size: int = 256,
    overlap: int = 32,
    in_channels: int = 9,
    device: str | None = None,
) -> np.ndarray:
    """Voorspel dijksegmentatie op een DTM-raster met sliding window.

    Parameters
    ----------
    dtm_path : str | Path
        Pad naar het DTM GeoTIFF.
    model_path : str | Path
        Pad naar het opgeslagen model (.pt).
    output_path : str | Path
        Pad voor het output GeoTIFF met klasselabels.
    rgb_path : str | Path | None
        Optioneel pad naar luchtfoto GeoTIFF (RGB).
    dsm_path : str | Path | None
        Optioneel pad naar DSM GeoTIFF (voor nDSM berekening).
    tile_size : int
        Grootte van de sliding window.
    overlap : int
        Overlap in pixels tussen tiles.
    in_channels : int
        Aantal inputkanalen (moet overeenkomen met training).
    device : str | None
        'cuda', 'cpu', of None (auto-detect).

    Returns
    -------
    np.ndarray
        Klasselabel-array (zelfde dimensies als input DTM).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Laad model
    model = build_unet(in_channels=in_channels, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # Lees DTM
    with rasterio.open(dtm_path) as src:
        dtm = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    dtm = np.nan_to_num(dtm, nan=0.0)
    h, w = dtm.shape

    # Lees RGB indien beschikbaar
    rgb = None
    if rgb_path is not None:
        rgb_path = Path(rgb_path)
        if rgb_path.exists():
            with rasterio.open(rgb_path) as src:
                rgb = src.read().astype(np.float32)  # (3, H, W)

    # Lees DSM indien beschikbaar
    dsm = None
    if dsm_path is not None:
        dsm_path = Path(dsm_path)
        if dsm_path.exists():
            with rasterio.open(dsm_path) as src:
                dsm = src.read(1).astype(np.float32)
            dsm = np.nan_to_num(dsm, nan=0.0)

    # Resultaat-arrays
    prediction = np.zeros((NUM_CLASSES, h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    step = tile_size - overlap

    with torch.no_grad():
        for y in range(0, h, step):
            for x in range(0, w, step):
                ye = min(y + tile_size, h)
                xe = min(x + tile_size, w)
                ys = ye - tile_size
                xs = xe - tile_size

                tile = dtm[ys:ye, xs:xe]
                channels = _build_channels(
                    tile, rgb, dsm, dtm, ys, ye, xs, xe, in_channels
                )

                inp = np.stack(channels, axis=0)[np.newaxis]
                inp_t = torch.from_numpy(inp).to(device)
                out = model(inp_t).cpu().numpy()[0]  # (C, H, W)

                prediction[:, ys:ye, xs:xe] += out
                counts[ys:ye, xs:xe] += 1

    # Gemiddelde en argmax
    counts[counts == 0] = 1
    prediction /= counts[np.newaxis]
    labels = prediction.argmax(axis=0).astype(np.uint8)

    # Post-processing: morfologische cleanup
    labels = _postprocess(labels)

    # Correctie binnen/buiten oriëntatie op basis van DTM hoogte
    labels = _correct_orientation(labels, dtm)

    # Schrijf output
    profile.update(dtype="uint8", count=1, nodata=0)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(labels, 1)

    print(f"Voorspelling opgeslagen: {output_path}")
    return labels


def _build_channels(
    tile: np.ndarray,
    rgb: np.ndarray | None,
    dsm: np.ndarray | None,
    dtm_full: np.ndarray,
    ys: int, ye: int, xs: int, xe: int,
    in_channels: int,
) -> list[np.ndarray]:
    """Bouw de inputkanalen voor een enkele tile.

    Kanaalvolgorde (9 kanalen default):
    DTM, slope, aspect, curvature, TPI, nDSM, R, G, B
    """
    channels = [_normalize(tile)]

    # Terrein-afgeleiden
    if in_channels >= 2:
        channels.append(_normalize(_compute_slope(tile)))
    if in_channels >= 3:
        channels.append(_normalize(_compute_aspect(tile)))
    if in_channels >= 4:
        channels.append(_normalize(_compute_curvature(tile)))
    if in_channels >= 5:
        channels.append(_normalize(_compute_tpi(tile)))

    # nDSM (DSM - DTM)
    if in_channels >= 6 and dsm is not None:
        dsm_tile = dsm[ys:ye, xs:xe]
        ndsm = np.clip(dsm_tile - tile, 0, 50)
        channels.append(_normalize(ndsm))
    elif in_channels >= 6 and dsm is None and rgb is not None:
        # nDSM niet beschikbaar maar wel verwacht — vul met zeros, schuif door naar RGB
        channels.append(np.zeros_like(tile))

    # RGB luchtfoto
    if rgb is not None:
        for band in range(rgb.shape[0]):
            channels.append(_normalize(rgb[band, ys:ye, xs:xe]))

    # Pad indien nodig tot in_channels
    while len(channels) < in_channels:
        channels.append(np.zeros_like(tile))

    return channels[:in_channels]


def _postprocess(labels: np.ndarray, min_area: int = 200) -> np.ndarray:
    """Morfologische cleanup van voorspelde labels.

    - Verwijder kleine geïsoleerde gebieden
    - Sluit kleine gaten via closing
    - Ruimtelijke constraint: insteek/sloot alleen naast dijkklassen
    - Verwijder insteek/sloot die > max_fraction van beeld beslaat
    """
    from scipy.ndimage import binary_opening, binary_closing, binary_dilation
    from scipy.ndimage import label as nd_label

    cleaned = labels.copy()

    # Stap 1: basis cleanup per klasse
    for cls in range(1, NUM_CLASSES):
        mask = (cleaned == cls)
        if mask.sum() == 0:
            continue

        # Insteek/sloot: grotere min_area (ze vormen snel grote blobs)
        cls_min_area = min_area * 5 if cls in (8, 9) else min_area

        mask = binary_closing(mask, iterations=2)
        mask = binary_opening(mask, iterations=1)

        labeled, n_components = nd_label(mask)
        for comp_id in range(1, n_components + 1):
            comp_mask = (labeled == comp_id)
            if comp_mask.sum() < cls_min_area:
                mask[comp_mask] = False

        cleaned[mask & (cleaned == 0)] = cls
        cleaned[~mask & (cleaned == cls)] = 0

    # Stap 2: ruimtelijke constraint — insteek/sloot moet grenzen aan dijkklassen
    # Dijkklassen: kruin(1), talud(2,5), teen(4,7), berm(3,6)
    dijk_mask = np.isin(cleaned, [1, 2, 3, 4, 5, 6, 7])
    if dijk_mask.any():
        # Dijkzone + buffer van 50px (~25m)
        dijk_zone = binary_dilation(dijk_mask, iterations=50)

        # Verwijder insteek/sloot pixels buiten de dijkzone
        for cls in (8, 9):
            outside = (cleaned == cls) & ~dijk_zone
            if outside.any():
                cleaned[outside] = 0

    # Stap 3: als insteek nog steeds > 30% is, verklein tot alleen nabij teen/buitenzijde
    total = cleaned.size
    insteek_frac = (cleaned == 8).sum() / total
    if insteek_frac > 0.25:
        # Alleen insteek behouden binnen 30px van teen_buiten of buitenteen
        outer_mask = np.isin(cleaned, [5, 6, 7])
        if outer_mask.any():
            outer_zone = binary_dilation(outer_mask, iterations=30)
            cleaned[(cleaned == 8) & ~outer_zone] = 0

    return cleaned


def _correct_orientation(labels: np.ndarray, dtm: np.ndarray) -> np.ndarray:
    """Corrigeer binnen/buiten oriëntatie op basis van DTM hoogte.

    Strategie: vergelijk de gemiddelde DTM-hoogte van de 'binnen'-kant
    (talud_binnen=2, teen_binnen=4, binnenberm=3) met de 'buiten'-kant
    (talud_buiten=5, teen_buiten=7, buitenberm=6). De buitenzijde
    (waterzijde) hoort lager te liggen. Als dat niet zo is, swap de klassen.

    Parameters
    ----------
    labels : np.ndarray
        Voorspelde klasselabels.
    dtm : np.ndarray
        DTM hoogtedata (zelfde shape als labels).

    Returns
    -------
    np.ndarray
        Gecorrigeerde labels.
    """
    binnen_mask = np.isin(labels, [2, 3, 4])  # talud_bi, berm_bi, teen_bi
    buiten_mask = np.isin(labels, [5, 6, 7])  # talud_bu, berm_bu, teen_bu

    # Alleen corrigeren als er voldoende pixels zijn van beide kanten
    if binnen_mask.sum() < 100 or buiten_mask.sum() < 100:
        return labels

    # Gemiddelde DTM hoogte per zijde (excl. nodata en extreme waarden)
    dtm_valid = dtm.copy().astype(np.float64)
    dtm_valid[(dtm_valid == 0) | (dtm_valid < -10) | (dtm_valid > 100)] = np.nan

    binnen_z = np.nanmean(dtm_valid[binnen_mask])
    buiten_z = np.nanmean(dtm_valid[buiten_mask])

    if np.isnan(binnen_z) or np.isnan(buiten_z):
        return labels

    # Buiten (waterkant) hoort lager te zijn dan binnen (polder)
    # Als buiten hoger is dan binnen → swap
    if buiten_z > binnen_z + 0.2:  # 0.2m marge
        print(f"  Orientatie-correctie: buiten ({buiten_z:.1f}m) > binnen ({binnen_z:.1f}m) -> swap")
        swap_map = {
            2: 5, 5: 2,  # talud_binnen ↔ talud_buiten
            3: 6, 6: 3,  # binnenberm ↔ buitenberm
            4: 7, 7: 4,  # teen_binnen ↔ teen_buiten
        }
        corrected = labels.copy()
        for old_cls, new_cls in swap_map.items():
            corrected[labels == old_cls] = new_cls
        return corrected
    else:
        print(f"  Orientatie OK: buiten ({buiten_z:.1f}m) <= binnen ({binnen_z:.1f}m)")

    return labels
