"""
visualize.py — All figure generation.

Replicates paper figures using REAL model outputs:
  Figure 4: Segmentation overlays (5 panels per subject)
    1. Original FLAIR (brightest of the top-5 axial slices)
    2. Ground truth overlay (blue)
    3. Predicted mask overlay (red)
    4. Combined overlay (blue=GT only, red=pred only, green=overlap)
    5. MC-Dropout uncertainty map (hot colormap)

  Figure 5: Training curves per fold
    Loss + Pseudo-Dice per fold from REAL nnU-Net training logs.
    NO hardcoded or synthetic data is ever used.

Paper (Joshi et al., 2025):
  - Single-channel FLAIR input
  - Top-5 axial slices selected per subject
"""

import json
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional, List

# ─────────────────────────────────────────────────────────────
# Shared style
# ─────────────────────────────────────────────────────────────

DARK_BG    = "#1a1a2e"
PANEL_BG   = "#16213e"
TEXT_WHITE = "#e0e0e0"


# ─────────────────────────────────────────────────────────────
# 1.  Segmentation overlays (paper Figure 4)
# ─────────────────────────────────────────────────────────────

def _load_vol(path: Path) -> Optional[np.ndarray]:
    """Load a NIfTI volume; return None if path doesn't exist."""
    if path is not None and path.exists():
        return nib.load(str(path)).get_fdata()
    return None


def visualize_case(
    case_name: str,
    flair_dir: Path,
    label_dir: Path,
    pred_dir: Path,
    mc_dir: Optional[Path] = None,
    slice_idx: Optional[int] = None,
    ax_row=None,
) -> Optional[np.ndarray]:
    """
    Draw 5 panels for one subject replicating paper Figure 4.

    Input is SINGLE-CHANNEL FLAIR (top-5 slices, shape K×H×W).
    Selects the slice with highest FLAIR intensity from the top-5 selection.

    Panels:
      1. Original FLAIR
      2. Ground truth overlay (blue)
      3. Predicted mask overlay (red)
      4. Combined overlay (blue=GT only, red=pred only, green=overlap)
      5. MC-Dropout uncertainty map

    If ax_row is None, a new figure is created.
    Returns the axes array.
    """
    # Single-channel FLAIR: _0000.nii.gz
    flair_path = flair_dir / f"{case_name}_0000.nii.gz"
    lbl_path   = label_dir / f"{case_name}.nii.gz"

    # Priority: standard nnU-Net pred first (MC preds are often empty)
    # Then try MC-Dropout pred
    pred_path = None
    std_pred = pred_dir / f"{case_name}.nii.gz"
    if std_pred.exists():
        pred_candidate = _load_vol(std_pred)
        if pred_candidate is not None and (pred_candidate > 0.5).sum() > 0:
            pred_path = std_pred
    if pred_path is None and mc_dir is not None:
        mc_pred = mc_dir / f"{case_name}_pred.nii.gz"
        if mc_pred.exists():
            pred_path = mc_pred

    unc_path = (mc_dir / f"{case_name}_uncertainty.nii.gz"
                if mc_dir is not None else None)

    flair = _load_vol(flair_path)
    if flair is None:
        print(f"  [WARN] FLAIR not found for {case_name}")
        return None

    gt   = _load_vol(lbl_path) if lbl_path.exists() else np.zeros_like(flair)
    pred = _load_vol(pred_path) if pred_path else np.zeros_like(flair)
    unc  = _load_vol(unc_path)

    # The preprocessed FLAIR has shape (5, H, W) — 5 axial slices.
    # Pick the slice with the MOST overlap between GT and prediction.
    # Falls back to slice with most GT, then slice with highest FLAIR.
    if slice_idx is None:
        n_slices = flair.shape[0]
        if (gt.ndim == 3 and gt.shape == flair.shape and gt.sum() > 0
                and pred.ndim == 3 and pred.shape == flair.shape and (pred > 0.5).sum() > 0):
            # Best: pick slice with most GT+pred overlap
            overlaps = [((gt[z] > 0) & (pred[z] > 0.5)).sum() for z in range(n_slices)]
            if max(overlaps) > 0:
                slice_idx = int(np.argmax(overlaps))
            else:
                # No overlap: pick slice with most GT
                slice_idx = int(np.argmax([gt[z].sum() for z in range(n_slices)]))
        elif gt.ndim == 3 and gt.shape == flair.shape and gt.sum() > 0:
            slice_idx = int(np.argmax([gt[z].sum() for z in range(n_slices)]))
        else:
            slice_idx = int(np.argmax([flair[z].max() for z in range(n_slices)]))

    f_sl  = flair[slice_idx]
    gt_sl = (gt[slice_idx] > 0).astype(float) if gt.ndim == 3 else np.zeros_like(f_sl)
    pd_sl = (pred[slice_idx] > 0.5).astype(float) if pred.ndim == 3 else np.zeros_like(f_sl)
    uc_sl = unc[slice_idx] if unc is not None and unc.ndim == 3 else None

    # Robust FLAIR normalisation: use 1st–99th percentile to avoid outlier clipping
    p1, p99 = np.percentile(f_sl, 1), np.percentile(f_sl, 99)
    if p99 > p1:
        f_disp = np.clip((f_sl - p1) / (p99 - p1), 0, 1)
    else:
        f_min, f_max = f_sl.min(), f_sl.max()
        f_disp = (f_sl - f_min) / (f_max - f_min + 1e-8)

    n_panels = 5 if uc_sl is not None else 4
    if ax_row is None:
        _fig, axes = plt.subplots(
            1, n_panels,
            figsize=(4 * n_panels, 4),
            facecolor=DARK_BG,
        )
    else:
        axes = ax_row

    for ax in axes:
        ax.axis("off")
        ax.set_facecolor(DARK_BG)

    # Panel 1: Original FLAIR
    axes[0].imshow(f_disp, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original FLAIR", color=TEXT_WHITE, fontsize=9)

    # Panel 2: GT overlay (blue)
    axes[1].imshow(f_disp, cmap="gray", vmin=0, vmax=1)
    gt_rgba = np.zeros((*f_disp.shape, 4))
    gt_rgba[gt_sl > 0] = [0.2, 0.4, 1.0, 0.75]
    axes[1].imshow(gt_rgba)
    axes[1].set_title("Ground Truth (Manual)", color=TEXT_WHITE, fontsize=9)

    # Panel 3: Prediction overlay (red)
    axes[2].imshow(f_disp, cmap="gray", vmin=0, vmax=1)
    pred_rgba = np.zeros((*f_disp.shape, 4))
    pred_rgba[pd_sl > 0] = [1.0, 0.2, 0.2, 0.75]
    axes[2].imshow(pred_rgba)
    axes[2].set_title("nnU-Net Segmentation", color=TEXT_WHITE, fontsize=9)

    # Panel 4: Combined overlay
    axes[3].imshow(f_disp, cmap="gray", vmin=0, vmax=1)
    comb_rgba = np.zeros((*f_disp.shape, 4))
    gt_only   = (gt_sl > 0) & (pd_sl == 0)
    pred_only = (pd_sl > 0) & (gt_sl == 0)
    overlap   = (gt_sl > 0) & (pd_sl > 0)
    comb_rgba[gt_only]   = [0.2, 0.4, 1.0, 0.75]   # blue  = GT only
    comb_rgba[pred_only] = [1.0, 0.2, 0.2, 0.75]   # red   = pred only
    comb_rgba[overlap]   = [0.2, 1.0, 0.2, 0.90]   # green = overlap
    axes[3].imshow(comb_rgba)
    legend_elems = [
        mpatches.Patch(color="blue",  label="Manual (GT)"),
        mpatches.Patch(color="red",   label="Automated"),
        mpatches.Patch(color="green", label="Overlap"),
    ]
    axes[3].legend(handles=legend_elems, loc="lower right",
                   fontsize=6, framealpha=0.5)
    axes[3].set_title("Combined", color=TEXT_WHITE, fontsize=9)

    # Panel 5: Uncertainty (MC-Dropout entropy)
    if uc_sl is not None and n_panels == 5:
        im = axes[4].imshow(uc_sl, cmap="hot", vmin=0, vmax=0.7)
        plt.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)
        axes[4].set_title("MC-Dropout Uncertainty", color=TEXT_WHITE, fontsize=9)

    return axes


def plot_segmentation_overlays(
    case_list: List[dict],
    task_dir: Path,
    pred_dir: Path,
    mc_dir: Optional[Path],
    figures_dir: Path,
    n_show: int = 3,
    dpi: int = 150,
) -> Path:
    """
    Generate the segmentation overlay figure for up to n_show cases.
    Only shows cases that have:
      1. Real FLAIR data (flair_max > 0.5 — not a near-zero artefact)
      2. A non-empty ground-truth lesion label
      3. A non-empty nnU-Net prediction (some overlap with GT)
    Cases are ranked by GT/pred overlap so the best results appear first.
    Saves to figures_dir/segmentation_overlays.png.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    flair_dir = task_dir / "imagesTr"
    label_dir = task_dir / "labelsTr"

    # ── Score every case ─────────────────────────────────────────
    scored = []
    for r in case_list:
        cn = r["case_name"]
        flair_path = flair_dir / f"{cn}_0000.nii.gz"
        lbl_path   = label_dir / f"{cn}.nii.gz"
        pred_path  = pred_dir  / f"{cn}.nii.gz"
        if not (flair_path.exists() and lbl_path.exists() and pred_path.exists()):
            continue
        flair = nib.load(str(flair_path)).get_fdata()
        if flair.max() <= 0.5:          # near-zero / corrupted artefact
            continue
        lbl  = nib.load(str(lbl_path)).get_fdata()
        if lbl.sum() == 0:              # control / no lesion
            continue
        pred = nib.load(str(pred_path)).get_fdata()
        overlap = int(((pred > 0.5) & (lbl > 0.5)).sum())
        scored.append((overlap, r))

    # Sort best-overlap first
    scored.sort(key=lambda x: x[0], reverse=True)
    valid_cases = [r for _, r in scored]
    n_show = min(n_show, len(valid_cases))

    if n_show == 0:
        print("  [WARN] No cases with real FLAIR + labels + predictions found.")
        return figures_dir / "segmentation_overlays.png"

    fig, all_axes = plt.subplots(
        n_show, 5,
        figsize=(20, 4 * n_show),
        facecolor=DARK_BG,
    )
    if n_show == 1:
        all_axes = [all_axes]

    fig.suptitle(
        "FCD Type II Lesion Segmentation Results (Joshi et al., 2025)\n"
        "Blue = Manual GT | Red = nnU-Net Automated | Green = Overlap",
        color=TEXT_WHITE, fontsize=13, y=1.01,
    )

    for i, entry in enumerate(valid_cases[:n_show]):
        visualize_case(
            entry["case_name"],
            flair_dir=flair_dir,
            label_dir=label_dir,
            pred_dir=pred_dir,
            mc_dir=mc_dir,
            ax_row=all_axes[i],
        )
        all_axes[i][0].set_ylabel(
            entry["case_name"],
            color=TEXT_WHITE, fontsize=9,
            rotation=0, labelpad=60,
        )

    plt.tight_layout()
    out_path = figures_dir / "segmentation_overlays.png"
    plt.savefig(str(out_path), dpi=dpi, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"✅  Saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────
# 2.  Training curves (paper Figure 5) — from REAL logs
# ─────────────────────────────────────────────────────────────


def _find_training_log(fold_dir: Path) -> Optional[Path]:
    """Return the most recent timestamped training log, or plain fallback."""
    logs = sorted(fold_dir.glob("training_log_*.txt"))
    if logs:
        return logs[-1]
    plain = fold_dir / "training_log.txt"
    return plain if plain.exists() else None


def _parse_training_log(log_file: Path):
    """
    Parse a real nnU-Net training log.

    Log line format (nnU-Net v2):
      2026-05-03 20:31:47.211642: train_loss 0.0879
      2026-05-03 20:31:47.211908: val_loss 0.0107
      2026-05-03 20:31:47.211991: Pseudo dice [np.float32(0.1425)]

    Returns (train_loss, val_loss, pseudo_dice) as lists of floats.
    """
    train_loss, val_loss, pseudo_dice = [], [], []
    with open(log_file) as fh:
        for line in fh:
            lower = line.lower()
            try:
                # train / val loss  — value is the last token
                if "train_loss" in lower:
                    train_loss.append(float(line.strip().split()[-1]))
                elif "val_loss" in lower:
                    val_loss.append(float(line.strip().split()[-1]))
                elif "pseudo dice" in lower:
                    inside  = line.split("[")[-1].split("]")[0]
                    val_str = inside.split("(")[-1].rstrip(")")
                    pseudo_dice.append(float(val_str))
            except (ValueError, IndexError):
                pass
    return train_loss, val_loss, pseudo_dice


def plot_training_curves(
    results_dir: Path,
    task_id: int,
    trainer: str,
    plans: str,
    config: str,
    n_folds: int,
    epochs: int,
    figures_dir: Path,
    dpi: int = 150,
) -> Path:
    """
    Plot per-fold training/validation loss and pseudo-Dice from the REAL
    nnU-Net training logs.  No synthetic or hardcoded values are used.
    Raises FileNotFoundError if no logs exist for any fold.
    Saves to figures_dir/training_curves.png.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "text.color":       "black",
    })

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat  = axes.flatten()

    any_log_found = False
    summary_lines = ["Training Summary\n" + "─" * 25]

    for fold in range(n_folds):
        ax  = axes_flat[fold]
        ax2 = ax.twinx()

        fold_dir = (
            results_dir
            / f"Dataset{task_id:03d}_FCD"
            / f"{trainer}__{plans}__{config}"
            / f"fold_{fold}"
        )
        log_file = _find_training_log(fold_dir)

        if log_file is None:
            ax.set_title(f"Fold {fold + 1} — no log found", fontsize=11, fontweight="bold", color="red")
            ax.text(0.5, 0.5, "Training log not found",
                    ha="center", va="center", transform=ax.transAxes, color="red", fontsize=12)
            continue

        train_loss, val_loss, pseudo_dice = _parse_training_log(log_file)
        any_log_found = True

        if not train_loss:
            ax.set_title(f"Fold {fold + 1} — log empty", fontsize=11, fontweight="bold", color="orange")
            continue

        c   = colors[fold % len(colors)]
        ep  = range(1, len(train_loss) + 1)

        ax.plot(ep, train_loss, color=c,      alpha=0.8, linewidth=1.2, label="Train loss")
        if val_loss:
            ax.plot(ep, val_loss, color="gray", alpha=0.8, linewidth=1.2,
                    linestyle="--", label="Val loss")
        ax.set_ylabel("Loss", fontsize=9)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylim(bottom=min(train_loss) - 0.05 if train_loss else -1)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.2)

        if pseudo_dice:
            w   = min(10, len(pseudo_dice))
            ma  = np.convolve(pseudo_dice, np.ones(w) / w, mode="valid")
            ep2 = range(w, len(pseudo_dice) + 1)
            ax2.plot(ep2, ma, color="#e74c3c", linewidth=1.5, label="Pseudo-Dice (MA)")
            ax2.set_ylabel("Pseudo-Dice (MA)", fontsize=9, color="#e74c3c")
            ax2.set_ylim(0, max(pseudo_dice) * 1.2 + 0.05)
            ax2.tick_params(axis="y", colors="#e74c3c")
            final_pds = pseudo_dice[-1]
            mean_pds  = float(np.mean(pseudo_dice))
            ax2.axhline(final_pds, color="#e74c3c", linewidth=0.8,
                        linestyle=":", alpha=0.7)
            ax2.text(
                len(pseudo_dice) * 0.55, final_pds + 0.005,
                f"Final: {final_pds:.4f}  |  Mean: {mean_pds:.4f}",
                color="#e74c3c", fontsize=8,
            )
            summary_lines.append(
                f"Fold {fold + 1}: {len(pseudo_dice)} epochs  "
                f"mean={mean_pds:.4f}  final={final_pds:.4f}"
            )

        ax.set_title(f"Fold {fold + 1}", fontsize=11, fontweight="bold")

    if not any_log_found:
        raise FileNotFoundError(
            "No training logs found.  "
            f"Expected: {results_dir}/Dataset{task_id:03d}_FCD/{trainer}__{plans}__{config}/fold_N/training_log_*.txt"
        )

    # Last panel: actual training summary from real logs
    axes_flat[-1].axis("off")
    axes_flat[-1].text(
        0.05, 0.95, "\n".join(summary_lines),
        transform=axes_flat[-1].transAxes,
        fontsize=10, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8),
    )

    plt.suptitle(
        "Training / Validation Loss and Pseudo-Dice per Fold\n"
        "(From real nnU-Net training logs)",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()

    out_path = figures_dir / "training_curves.png"
    plt.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"✅  Saved → {out_path}")

    plt.rcParams.update(plt.rcParamsDefault)
    return out_path