"""Training loop voor het dijk-segmentatiemodel."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .dataset import DikeTileDataset, NUM_CLASSES
from .model import build_unet


def train_model(
    tiles_dir: str | Path,
    labels_dir: str | Path,
    output_dir: str | Path = "models/checkpoints",
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    val_split: float = 0.2,
    include_slope: bool = True,
    device: str | None = None,
) -> nn.Module:
    """Train het U-Net segmentatiemodel.

    Parameters
    ----------
    tiles_dir, labels_dir : str | Path
        Mappen met DTM-tiles en corresponderende labels.
    output_dir : str | Path
        Map om model-checkpoints op te slaan.
    epochs : int
        Aantal trainingsepochs.
    batch_size : int
        Batchgrootte.
    lr : float
        Learning rate.
    val_split : float
        Fractie van data voor validatie.
    include_slope : bool
        Slope-kanaal meenemen als input.
    device : str | None
        'cuda', 'cpu', of None (auto-detect).

    Returns
    -------
    nn.Module
        Het getrainde model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    dataset = DikeTileDataset(
        tiles_dir, labels_dir, include_slope=include_slope, augment=True
    )
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = build_unet(in_channels=dataset.num_channels, num_classes=NUM_CLASSES)
    model = model.to(device)

    # Class weights — kruin-klasse (1) extra gewicht geven
    weights = torch.ones(NUM_CLASSES, device=device)
    weights[1] = 3.0  # kruin
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validatie
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()
        val_loss /= max(len(val_loader), 1)

        scheduler.step()

        print(f"Epoch {epoch+1:3d}/{epochs} — train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}")

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    # Laatste model ook opslaan
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    print(f"Training voltooid. Beste val_loss: {best_val_loss:.4f}")
    return model
