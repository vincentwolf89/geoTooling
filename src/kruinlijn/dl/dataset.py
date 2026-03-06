"""Dataset voor dijk-segmentatie op basis van DTM-tiles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


# Klasse-labels voor segmentatie
CLASSES = {
    0: "achtergrond",
    1: "kruin",
    2: "talud_binnen",
    3: "teen_binnen",
    4: "talud_buiten",
    5: "teen_buiten",
}
NUM_CLASSES = len(CLASSES)


class DikeTileDataset(Dataset):
    """Dataset dat DTM-tiles en bijbehorende label-masks laadt.

    Parameters
    ----------
    tiles_dir : str | Path
        Map met DTM GeoTIFF-tiles.
    labels_dir : str | Path
        Map met label GeoTIFF-tiles (zelfde bestandsnamen als tiles_dir).
        Pixelwaarden 0..5 conform ``CLASSES``.
    tile_size : int
        Verwachte tilegrootte in pixels (tiles worden gecheckt).
    include_slope : bool
        Voeg een slope-kanaal toe naast het hoogtekanaal.
    include_aspect : bool
        Voeg een aspect-kanaal toe.
    augment : bool
        Pas data-augmentatie toe (flips, rotaties).
    """

    def __init__(
        self,
        tiles_dir: str | Path,
        labels_dir: str | Path,
        tile_size: int = 256,
        include_slope: bool = True,
        include_aspect: bool = False,
        augment: bool = False,
    ):
        self.tiles_dir = Path(tiles_dir)
        self.labels_dir = Path(labels_dir)
        self.tile_size = tile_size
        self.include_slope = include_slope
        self.include_aspect = include_aspect
        self.augment = augment

        self.tile_files = sorted(self.tiles_dir.glob("*.tif"))
        if not self.tile_files:
            raise FileNotFoundError(f"Geen tiles gevonden in {tiles_dir}")

    def __len__(self) -> int:
        return len(self.tile_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tile_path = self.tile_files[idx]
        label_path = self.labels_dir / tile_path.name

        # Lees DTM tile
        with rasterio.open(tile_path) as src:
            dtm = src.read(1).astype(np.float32)

        # Lees label mask
        with rasterio.open(label_path) as src:
            mask = src.read(1).astype(np.int64)

        # Vervang nodata
        dtm = np.nan_to_num(dtm, nan=0.0)

        # Bouw input-kanalen
        channels = [_normalize(dtm)]
        if self.include_slope:
            channels.append(_normalize(_compute_slope(dtm)))
        if self.include_aspect:
            channels.append(_normalize(_compute_aspect(dtm)))

        image = np.stack(channels, axis=0)

        # Augmentatie
        if self.augment:
            image, mask = _augment(image, mask)

        return torch.from_numpy(image), torch.from_numpy(mask)

    @property
    def num_channels(self) -> int:
        n = 1
        if self.include_slope:
            n += 1
        if self.include_aspect:
            n += 1
        return n


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalisatie naar [0, 1]."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _compute_slope(dtm: np.ndarray) -> np.ndarray:
    """Bereken slope (helling) in graden."""
    dy, dx = np.gradient(dtm)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    return np.degrees(slope)


def _compute_aspect(dtm: np.ndarray) -> np.ndarray:
    """Bereken aspect (expositie) in graden 0-360."""
    dy, dx = np.gradient(dtm)
    aspect = np.degrees(np.arctan2(-dy, dx))
    aspect[aspect < 0] += 360
    return aspect


def _augment(
    image: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Eenvoudige augmentatie: random flips en 90-graden rotaties."""
    # Random horizontale flip
    if np.random.rand() > 0.5:
        image = image[:, :, ::-1].copy()
        mask = mask[:, ::-1].copy()
    # Random verticale flip
    if np.random.rand() > 0.5:
        image = image[:, ::-1, :].copy()
        mask = mask[::-1, :].copy()
    # Random 90-graden rotatie
    k = np.random.randint(0, 4)
    if k > 0:
        image = np.rot90(image, k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k, axes=(0, 1)).copy()
    return image, mask
