"""Demo: hybride pipeline — morfologische labels genereren, dan DL model trainen.

Gebruik:
    python examples/demo_hybrid.py <dtm.tif> <hartlijn.gpkg> <output_dir/>

Stappen:
    1. Morfologische analyse -> kruinlijn + automatische labels
    2. Tiles knippen uit DTM + labels
    3. U-Net trainen op de gegenereerde labels
    4. Voorspelling doen op het volledige DTM
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
import geopandas as gpd
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kruinlijn.pipeline import morpho_pipeline, generate_training_labels


def cut_tiles(
    dtm_path: str,
    labels_path: str,
    tiles_dir: Path,
    labels_dir: Path,
    tile_size: int = 256,
    stride: int = 128,
):
    """Knip DTM en labels in tiles voor training."""
    tiles_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dtm_path) as dtm_src, rasterio.open(labels_path) as lbl_src:
        h, w = dtm_src.height, dtm_src.width
        count = 0

        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                window = Window(x, y, tile_size, tile_size)
                dtm_tile = dtm_src.read(1, window=window)
                lbl_tile = lbl_src.read(1, window=window)

                # Skip lege tiles (alleen achtergrond)
                if lbl_tile.max() == 0:
                    continue

                tile_name = f"tile_{count:05d}.tif"
                tile_transform = dtm_src.window_transform(window)
                profile = dtm_src.profile.copy()
                profile.update(width=tile_size, height=tile_size, transform=tile_transform)

                # Schrijf DTM tile
                with rasterio.open(tiles_dir / tile_name, "w", **profile) as dst:
                    dst.write(dtm_tile, 1)

                # Schrijf label tile
                lbl_profile = profile.copy()
                lbl_profile.update(dtype="uint8", nodata=0)
                with rasterio.open(labels_dir / tile_name, "w", **lbl_profile) as dst:
                    dst.write(lbl_tile.astype(np.uint8), 1)

                count += 1

    print(f"{count} tiles gegenereerd in {tiles_dir} en {labels_dir}")


def main():
    if len(sys.argv) < 4:
        print("Gebruik: python demo_hybrid.py <dtm.tif> <hartlijn.gpkg> <output_dir/>")
        sys.exit(1)

    dtm_path = sys.argv[1]
    hartlijn_path = sys.argv[2]
    output_dir = Path(sys.argv[3])
    output_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(hartlijn_path)
    centerline = gdf.geometry.iloc[0]

    # Stap 1: Genereer labels via morfologische methode
    print("=== Stap 1: Morfologische labels genereren ===")
    labels_path = output_dir / "morpho_labels.tif"
    generate_training_labels(dtm_path, centerline, labels_path)

    # Stap 2: Knip tiles
    print("\n=== Stap 2: Tiles knippen ===")
    tiles_dir = output_dir / "tiles" / "dtm"
    labels_dir = output_dir / "tiles" / "labels"
    cut_tiles(dtm_path, str(labels_path), tiles_dir, labels_dir)

    # Stap 3: Train model (optioneel — vereist torch)
    try:
        from kruinlijn.dl import train_model

        print("\n=== Stap 3: Model trainen ===")
        model = train_model(
            tiles_dir=tiles_dir,
            labels_dir=labels_dir,
            output_dir=output_dir / "checkpoints",
            epochs=20,
            batch_size=4,
        )

        # Stap 4: Voorspelling
        from kruinlijn.dl import predict_tiles

        print("\n=== Stap 4: Voorspelling ===")
        predict_tiles(
            dtm_path=dtm_path,
            model_path=output_dir / "checkpoints" / "best_model.pt",
            output_path=output_dir / "dl_prediction.tif",
        )
    except ImportError:
        print("\nPyTorch niet geinstalleerd — sla DL-stappen over.")
        print("Installeer met: pip install torch torchvision")

    print("\nKlaar!")


if __name__ == "__main__":
    main()
