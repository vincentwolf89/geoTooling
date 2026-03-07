"""Validatie van morfologische en DL resultaten.

Toont dwarsprofielen met knikpunten op echte AHN4 data,
en berekent metrics voor de DL voorspelling.

Gebruik:
    python examples/validate_results.py
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import LineString, shape

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kruinlijn.morpho import (
    generate_cross_profiles,
    extract_profile_elevations,
    detect_crest_points,
    detect_knikpunten,
)
from kruinlijn.dl.dataset import CLASSES

OUTPUT_DIR = Path("output/dl_waal")
DTM_PATH = OUTPUT_DIR / "dtm_waal_section.tif"
LABELS_PATH = OUTPUT_DIR / "labels_waal_section.tif"
PRED_PATH = OUTPUT_DIR / "prediction_waal.tif"
GEOJSON_PATH = Path("data/raw/Trajecten ZWO.geojson")


def plot_cross_profiles(centerline: LineString, dtm_path: str, n_profiles: int = 6):
    """Plot meerdere dwarsprofielen met alle knikpunten."""
    profiles = generate_cross_profiles(centerline, spacing=5.0, width=60.0)
    profiles = extract_profile_elevations(profiles, dtm_path)
    profiles = detect_crest_points(profiles, method="curvature")
    profiles = detect_knikpunten(profiles, smooth_sigma=3.0)

    # Kies profielen verspreid over de sectie
    valid = [p for p in profiles if p.get("crest_idx") is not None]
    if len(valid) < n_profiles:
        n_profiles = len(valid)
    step = max(1, len(valid) // n_profiles)
    selected = valid[::step][:n_profiles]

    cols = min(3, n_profiles)
    rows = (n_profiles + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
    if n_profiles == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    knikpunt_info = [
        ("crest", "Kruin", "red", "o", 12),
        ("binnenkruin", "Binnenkruin", "orange", "s", 10),
        ("buitenkruin", "Buitenkruin", "darkorange", "s", 10),
        ("binnenberm", "Binnenberm", "purple", "D", 9),
        ("buitenberm", "Buitenberm", "mediumpurple", "D", 9),
        ("binnenteen", "Binnenteen", "green", "^", 10),
        ("buitenteen", "Buitenteen", "blue", "^", 10),
    ]

    for i, p in enumerate(selected):
        ax = axes[i]
        offsets = p["offsets"]
        z = p["elevations"]

        # Hoogteprofiel
        ax.plot(offsets, z, "k-", linewidth=1.5, label="Hoogteprofiel")

        # Knikpunten
        for key, label, kleur, marker, ms in knikpunt_info:
            idx = p.get(f"{key}_idx")
            z_val = p.get(f"{key}_z")
            if idx is not None and z_val is not None:
                ax.plot(offsets[idx], z_val, color=kleur, marker=marker,
                        markersize=ms, zorder=5, label=label)
                ax.axvline(offsets[idx], color=kleur, linestyle=":", alpha=0.3)

        ax.set_xlabel("Afstand op profiel (m)")
        ax.set_ylabel("Hoogte (m NAP)")
        ax.set_title(f"Profiel op {p['distance']:.0f}m")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc="best")

    # Verberg lege subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = OUTPUT_DIR / "validatie_profielen.png"
    plt.savefig(path, dpi=150)
    print(f"Profielen opgeslagen: {path}")
    plt.close()


def compute_metrics():
    """Bereken per-klasse metrics voor DL voorspelling vs. morfologische labels."""
    if not PRED_PATH.exists():
        print("Geen DL voorspelling gevonden, sla metrics over.")
        return

    with rasterio.open(str(LABELS_PATH)) as src:
        labels = src.read(1).flatten()
    with rasterio.open(str(PRED_PATH)) as src:
        preds = src.read(1).flatten()

    # Alleen pixels waar labels > 0 OF preds > 0 (skip pure achtergrond)
    mask = (labels > 0) | (preds > 0)
    labels_m = labels[mask]
    preds_m = preds[mask]

    n_classes = max(labels.max(), preds.max()) + 1

    print("\n" + "=" * 70)
    print("VALIDATIE METRICS: DL voorspelling vs. morfologische labels")
    print("=" * 70)

    # Overall accuracy
    total = len(labels_m)
    correct = (labels_m == preds_m).sum()
    print(f"\nOverall accuracy (dijk-pixels): {correct/total:.1%} ({correct}/{total})")

    # Per-klasse metrics
    print(f"\n{'Klasse':<20s} {'Precision':>10s} {'Recall':>10s} {'IoU':>10s} {'Support':>10s}")
    print("-" * 62)

    ious = []
    for c in range(n_classes):
        name = CLASSES.get(c, f"klasse_{c}")
        tp = ((preds_m == c) & (labels_m == c)).sum()
        fp = ((preds_m == c) & (labels_m != c)).sum()
        fn = ((preds_m != c) & (labels_m == c)).sum()
        support = (labels_m == c).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

        if support > 0:
            ious.append(iou)

        print(f"{name:<20s} {precision:>10.1%} {recall:>10.1%} {iou:>10.1%} {support:>10d}")

    print("-" * 62)
    print(f"{'Mean IoU':<20s} {'':>10s} {'':>10s} {np.mean(ious):>10.1%}")

    # Confusion matrix als heatmap
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for c_true in range(n_classes):
        for c_pred in range(n_classes):
            cm[c_true, c_pred] = ((labels_m == c_true) & (preds_m == c_pred)).sum()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_xlabel("Voorspeld")
    ax.set_ylabel("Label (morfologisch)")
    ax.set_title("Confusion Matrix (dijk-pixels)")

    class_names = [CLASSES.get(i, f"?{i}") for i in range(n_classes)]
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    # Waarden in cellen
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm[i, j]
            if val > 0:
                color = "white" if val > cm.max() * 0.5 else "black"
                ax.text(j, i, f"{val}", ha="center", va="center",
                        fontsize=7, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    path = OUTPUT_DIR / "validatie_confusion_matrix.png"
    plt.savefig(path, dpi=150)
    print(f"\nConfusion matrix opgeslagen: {path}")
    plt.close()


def plot_comparison():
    """Zij-aan-zij vergelijking van labels en voorspelling, ingezoomd op de dijk."""
    if not PRED_PATH.exists():
        return

    with rasterio.open(str(DTM_PATH)) as src:
        dtm = src.read(1)
        transform = src.transform
    with rasterio.open(str(LABELS_PATH)) as src:
        labels = src.read(1)
    with rasterio.open(str(PRED_PATH)) as src:
        preds = src.read(1)

    # Zoek de rijen waar dijk-pixels zitten voor inzoomen
    dijk_rows = np.where(labels.max(axis=1) > 0)[0]
    if len(dijk_rows) == 0:
        return
    row_min = max(0, dijk_rows[0] - 20)
    row_max = min(labels.shape[0], dijk_rows[-1] + 20)

    labels_crop = labels[row_min:row_max, :]
    preds_crop = preds[row_min:row_max, :]
    dtm_crop = dtm[row_min:row_max, :]

    fig, axes = plt.subplots(3, 1, figsize=(18, 12))

    # DTM
    im0 = axes[0].imshow(dtm_crop, cmap="terrain", aspect="auto")
    axes[0].set_title("DTM (ingezoomd op dijkzone)")
    plt.colorbar(im0, ax=axes[0], shrink=0.6, label="m NAP")

    # Labels
    cmap = plt.cm.get_cmap("tab10", 7)
    im1 = axes[1].imshow(labels_crop, cmap=cmap, vmin=0, vmax=6, aspect="auto")
    axes[1].set_title("Morfologische labels")
    cbar1 = plt.colorbar(im1, ax=axes[1], shrink=0.6)
    cbar1.set_ticks(range(7))
    cbar1.set_ticklabels([CLASSES.get(i, f"?{i}") for i in range(7)])

    # Voorspelling
    im2 = axes[2].imshow(preds_crop, cmap=cmap, vmin=0, vmax=6, aspect="auto")
    axes[2].set_title("DL voorspelling")
    cbar2 = plt.colorbar(im2, ax=axes[2], shrink=0.6)
    cbar2.set_ticks(range(7))
    cbar2.set_ticklabels([CLASSES.get(i, f"?{i}") for i in range(7)])

    plt.tight_layout()
    path = OUTPUT_DIR / "validatie_vergelijking.png"
    plt.savefig(path, dpi=150)
    print(f"Vergelijking opgeslagen: {path}")
    plt.close()


def main():
    if not DTM_PATH.exists():
        print("Geen DTM gevonden. Draai eerst demo_dl_waal.py")
        return

    print("=" * 60)
    print("Validatie - Morfologische + DL resultaten")
    print("=" * 60)

    # Lees hartlijn
    with open(GEOJSON_PATH) as f:
        geojson = json.load(f)
    full_line = shape(geojson["features"][0]["geometry"])
    section_points = []
    d = 6000.0
    while d <= 6500.0:
        pt = full_line.interpolate(d)
        section_points.append((pt.x, pt.y))
        d += 2.0
    centerline = LineString(section_points)

    # 1. Dwarsprofielen
    print("\n[1/3] Dwarsprofielen met knikpunten...")
    plot_cross_profiles(centerline, str(DTM_PATH), n_profiles=6)

    # 2. Metrics
    print("\n[2/3] Berekenen metrics...")
    compute_metrics()

    # 3. Visuele vergelijking
    print("\n[3/3] Visuele vergelijking labels vs. voorspelling...")
    plot_comparison()

    print("\n" + "=" * 60)
    print("Validatie voltooid!")
    print(f"Alle output in: {OUTPUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
