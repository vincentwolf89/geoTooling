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
    3: "binnenberm",
    4: "teen_binnen",
    5: "talud_buiten",
    6: "buitenberm",
    7: "teen_buiten",
    8: "insteek",
    9: "sloot",
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
    rgb_dir : str | Path | None
        Optionele map met luchtfoto-tiles (RGB GeoTIFF, zelfde bestandsnamen).
    dsm_dir : str | Path | None
        Optionele map met DSM-tiles voor nDSM (DSM - DTM) kanaal.
    tile_size : int
        Verwachte tilegrootte in pixels.
    include_slope : bool
        Voeg een slope-kanaal toe naast het hoogtekanaal.
    include_aspect : bool
        Voeg een aspect-kanaal toe (hellingsrichting).
    include_curvature : bool
        Voeg een curvature-kanaal toe (2e afgeleide, markeert knikpunten).
    include_tpi : bool
        Voeg TPI-kanaal toe (Topographic Position Index, relatieve hoogte).
    augment : bool
        Pas data-augmentatie toe (flips, rotaties).
    """

    def __init__(
        self,
        tiles_dir: str | Path,
        labels_dir: str | Path,
        rgb_dir: str | Path | None = None,
        dsm_dir: str | Path | None = None,
        tile_size: int = 256,
        include_slope: bool = True,
        include_aspect: bool = True,
        include_curvature: bool = True,
        include_tpi: bool = True,
        augment: bool = False,
    ):
        self.tiles_dir = Path(tiles_dir)
        self.labels_dir = Path(labels_dir)
        self.rgb_dir = Path(rgb_dir) if rgb_dir else None
        self.dsm_dir = Path(dsm_dir) if dsm_dir else None
        self.tile_size = tile_size
        self.include_slope = include_slope
        self.include_aspect = include_aspect
        self.include_curvature = include_curvature
        self.include_tpi = include_tpi
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

        # Bouw input-kanalen: terrein-afgeleiden
        channels = [_normalize(dtm)]
        if self.include_slope:
            channels.append(_normalize(_compute_slope(dtm)))
        if self.include_aspect:
            channels.append(_normalize(_compute_aspect(dtm)))
        if self.include_curvature:
            channels.append(_normalize(_compute_curvature(dtm)))
        if self.include_tpi:
            channels.append(_normalize(_compute_tpi(dtm)))

        # nDSM kanaal (DSM - DTM = vegetatie/objecthoogte)
        h, w = dtm.shape
        if self.dsm_dir is not None:
            dsm_path = self.dsm_dir / tile_path.name
            if dsm_path.exists():
                with rasterio.open(dsm_path) as src:
                    dsm = src.read(1).astype(np.float32)
                dsm = np.nan_to_num(dsm, nan=0.0)
                ndsm = np.clip(dsm - dtm, 0, 50)
                channels.append(_normalize(ndsm))
            else:
                channels.append(np.zeros((h, w), dtype=np.float32))

        # Luchtfoto (RGB) kanalen
        if self.rgb_dir is not None:
            rgb_path = self.rgb_dir / tile_path.name
            if rgb_path.exists():
                with rasterio.open(rgb_path) as src:
                    for band in range(1, min(src.count, 3) + 1):
                        band_data = src.read(band).astype(np.float32)
                        channels.append(_normalize(band_data))
            else:
                for _ in range(3):
                    channels.append(np.zeros((h, w), dtype=np.float32))

        image = np.stack(channels, axis=0)

        # Augmentatie
        if self.augment:
            image, mask = _augment(image, mask)

        return torch.from_numpy(image), torch.from_numpy(mask)

    @property
    def num_channels(self) -> int:
        n = 1  # DTM
        if self.include_slope:
            n += 1
        if self.include_aspect:
            n += 1
        if self.include_curvature:
            n += 1
        if self.include_tpi:
            n += 1
        if self.dsm_dir is not None:
            n += 1  # nDSM
        if self.rgb_dir is not None:
            n += 3  # R, G, B
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


def _compute_curvature(dtm: np.ndarray) -> np.ndarray:
    """Bereken curvature (2e afgeleide) — markeert knikpunten in het profiel."""
    from scipy.ndimage import laplace
    curv = laplace(dtm)
    # Clip extreme waarden
    p99 = np.percentile(np.abs(curv), 99)
    return np.clip(curv, -p99, p99)


def _compute_tpi(dtm: np.ndarray, radius: int = 15) -> np.ndarray:
    """Bereken Topographic Position Index: hoogte t.o.v. omgeving.

    Positieve waarden = hoger dan omgeving (kruin), negatief = lager (sloot).
    """
    from scipy.ndimage import uniform_filter
    mean_elev = uniform_filter(dtm, size=radius)
    return dtm - mean_elev


def _augment(
    image: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Augmentatie: flips, rotaties, brightness/contrast jitter, noise."""
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

    # Brightness/contrast jitter per kanaal
    if np.random.rand() > 0.5:
        for c in range(image.shape[0]):
            brightness = np.random.uniform(-0.1, 0.1)
            contrast = np.random.uniform(0.85, 1.15)
            image[c] = np.clip(image[c] * contrast + brightness, 0, 1)

    # Gaussian noise
    if np.random.rand() > 0.7:
        sigma = np.random.uniform(0.01, 0.03)
        noise = np.random.randn(*image.shape).astype(np.float32) * sigma
        image = np.clip(image + noise, 0, 1)

    # Elastic-achtige deformatie via random affine shift per rij
    if np.random.rand() > 0.8:
        _, h, w = image.shape
        max_shift = max(1, w // 50)
        shifts = np.random.randint(-max_shift, max_shift + 1, size=h)
        # Smooth de shifts
        from scipy.ndimage import uniform_filter1d
        shifts = uniform_filter1d(shifts.astype(float), size=10).astype(int)
        for row in range(h):
            s = shifts[row]
            if s != 0:
                image[:, row, :] = np.roll(image[:, row, :], s, axis=-1)
                mask[row, :] = np.roll(mask[row, :], s, axis=-1)

    return image, mask
