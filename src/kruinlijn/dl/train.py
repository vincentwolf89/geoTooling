"""Training loop voor het dijk-segmentatiemodel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from .dataset import DikeTileDataset, NUM_CLASSES
from .model import build_unet


class FocalLoss(nn.Module):
    """Focal Loss voor klasse-imbalans: reduceert gewicht van makkelijke voorbeelden."""

    def __init__(self, weight=None, gamma: float = 2.0, ignore_index: int = -1):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.weight,
            ignore_index=self.ignore_index, reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss per klasse, gemiddeld over niet-lege klassen."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(inputs, dim=1)
        targets_oh = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()

        dice_per_class = []
        for c in range(inputs.shape[1]):
            p = probs[:, c]
            t = targets_oh[:, c]
            intersection = (p * t).sum()
            union = p.sum() + t.sum()
            if union > 0:
                dice_per_class.append(
                    1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
                )
        if not dice_per_class:
            return torch.tensor(0.0, device=inputs.device)
        return torch.stack(dice_per_class).mean()


def compute_class_weights(dataset: DikeTileDataset, device: str) -> torch.Tensor:
    """Bereken inverse-frequency class weights uit de dataset."""
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    # Sample een subset voor snelheid
    n_sample = min(len(dataset), 50)
    indices = np.random.choice(len(dataset), n_sample, replace=False)
    for idx in indices:
        _, mask = dataset[idx]
        for c in range(NUM_CLASSES):
            counts[c] += (mask.numpy() == c).sum()

    # Inverse frequency, geclipt
    total = counts.sum()
    weights = np.where(counts > 0, total / (NUM_CLASSES * counts), 1.0)
    # Cap maximum weight zodat zeldzame klassen niet exploderen
    weights = np.clip(weights, 0.5, 10.0)
    print(f"  Class weights: {dict(zip(range(NUM_CLASSES), weights.round(2)))}")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_model(
    tiles_dir: str | Path,
    labels_dir: str | Path,
    rgb_dir: str | Path | None = None,
    output_dir: str | Path = "models/checkpoints",
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    val_split: float = 0.2,
    include_slope: bool = True,
    device: str | None = None,
) -> nn.Module:
    """Train het Attention U-Net segmentatiemodel.

    Gebruikt Focal Loss + Dice Loss voor betere handling van klasse-imbalans,
    met automatische class weights en deep supervision.

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
        tiles_dir, labels_dir, rgb_dir=rgb_dir,
        include_slope=include_slope, augment=True,
    )
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = build_unet(in_channels=dataset.num_channels, num_classes=NUM_CLASSES)
    model = model.to(device)

    # Automatische class weights
    print("  Berekenen class weights...")
    weights = compute_class_weights(dataset, device)

    # Combined loss: focal + dice
    focal_loss = FocalLoss(weight=weights, gamma=2.0)
    dice_loss = DiceLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader),
    )

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 15

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            # Deep supervision: model returns (main, aux3, aux2) during training
            outputs = model(images)
            if isinstance(outputs, tuple):
                main_out, aux3, aux2 = outputs
                loss_main = focal_loss(main_out, masks) + dice_loss(main_out, masks)
                loss_aux3 = focal_loss(aux3, masks) + dice_loss(aux3, masks)
                loss_aux2 = focal_loss(aux2, masks) + dice_loss(aux2, masks)
                loss = loss_main + 0.3 * loss_aux3 + 0.2 * loss_aux2
            else:
                loss = focal_loss(outputs, masks) + dice_loss(outputs, masks)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validatie
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                vl = focal_loss(outputs, masks) + dice_loss(outputs, masks)
                val_loss += vl.item()
                # Accuracy op dijk-pixels
                preds = outputs.argmax(dim=1)
                dijk_mask = masks > 0
                if dijk_mask.any():
                    val_correct += (preds[dijk_mask] == masks[dijk_mask]).sum().item()
                    val_total += dijk_mask.sum().item()
        val_loss /= max(len(val_loader), 1)
        val_acc = val_correct / max(val_total, 1)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:3d}/{epochs} — "
            f"train: {train_loss:.4f}  val: {val_loss:.4f}  "
            f"acc: {val_acc:.1%}  lr: {current_lr:.2e}"
        )

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"  Early stopping na {patience} epochs zonder verbetering.")
            break

    # Laatste model ook opslaan
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    print(f"Training voltooid. Beste val_loss: {best_val_loss:.4f}")
    return model
