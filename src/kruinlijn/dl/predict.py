"""Voorspelling met een getraind segmentatiemodel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch

from .dataset import _normalize, _compute_slope, NUM_CLASSES
from .model import build_unet


def predict_tiles(
    dtm_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    rgb_path: str | Path | None = None,
    tile_size: int = 256,
    overlap: int = 32,
    in_channels: int = 2,
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
                channels = [_normalize(tile)]
                if in_channels >= 2:
                    channels.append(_normalize(_compute_slope(tile)))
                if rgb is not None:
                    for band in range(rgb.shape[0]):
                        channels.append(_normalize(rgb[band, ys:ye, xs:xe]))

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

    # Schrijf output
    profile.update(dtype="uint8", count=1, nodata=0)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(labels, 1)

    print(f"Voorspelling opgeslagen: {output_path}")
    return labels


def _postprocess(labels: np.ndarray, min_area: int = 100) -> np.ndarray:
    """Morfologische cleanup van voorspelde labels.

    - Verwijder kleine geïsoleerde gebieden (< min_area pixels)
    - Sluit kleine gaten via closing
    - Behoud ruimtelijke samenhang
    """
    from scipy.ndimage import binary_opening, binary_closing, label as nd_label

    cleaned = labels.copy()

    # Per klasse (niet achtergrond): verwijder kleine componenten
    for cls in range(1, NUM_CLASSES):
        mask = (cleaned == cls)
        if mask.sum() == 0:
            continue

        # Closing: vul kleine gaten
        mask = binary_closing(mask, iterations=2)
        # Opening: verwijder kleine ruis
        mask = binary_opening(mask, iterations=1)

        # Verwijder componenten kleiner dan min_area
        labeled, n_components = nd_label(mask)
        for comp_id in range(1, n_components + 1):
            comp_mask = (labeled == comp_id)
            if comp_mask.sum() < min_area:
                mask[comp_mask] = False

        # Schrijf terug
        cleaned[mask & (cleaned == 0)] = cls
        cleaned[~mask & (cleaned == cls)] = 0

    return cleaned
