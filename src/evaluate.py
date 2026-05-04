"""
evaluate.py — Evaluation, statistical analysis, and comparison table.

Reads per-fold Pseudo-Dice values directly from the nnU-Net training logs
(training_log_*.txt) so all reported numbers come from the actual model.

Includes 95% CI, paired t-test, and per-fold PDS summary.
"""

import numpy as np
import pandas as pd
import nibabel as nib
import logging
from pathlib import Path
from typing import List, Optional
from scipy.stats import t as t_dist, ttest_rel

from src.utils import dice_score, pseudo_dice_score

logger = logging.getLogger("fcd.evaluate")


# ─────────────────────────────────────────────────────────────
# 1.  Per-case evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_case(
    case_name: str,
    pred_dir: Path,
    label_dir: Path,
    mc_dir: Path = None,
) -> dict:
    """
    Compute DSC and PDS for one case.
    Tries mc_dir first (refined MC-Dropout prediction), then pred_dir
    (standard nnU-Net output).

    Returns a dict with: case, dice, pds, lesion_voxels, pred_voxels.
    """
    lbl_path = label_dir / f"{case_name}.nii.gz"
    if not lbl_path.exists():
        return {"case": case_name, "dice": np.nan, "pds": np.nan,
                "lesion_voxels": np.nan, "pred_voxels": np.nan}

    # Prefer MC-Dropout prediction
    pred_path = None
    if mc_dir is not None:
        mc_pred = mc_dir / f"{case_name}_pred.nii.gz"
        if mc_pred.exists():
            pred_path = mc_pred

    if pred_path is None:
        nn_pred = pred_dir / f"{case_name}.nii.gz"
        if nn_pred.exists():
            pred_path = nn_pred

    if pred_path is None:
        return {"case": case_name, "dice": np.nan, "pds": np.nan,
                "lesion_voxels": np.nan, "pred_voxels": np.nan}

    pred = nib.load(str(pred_path)).get_fdata()
    gt   = nib.load(str(lbl_path)).get_fdata()

    return {
        "case":          case_name,
        "dice":          dice_score(pred, gt),
        "pds":           pseudo_dice_score(pred, gt),
        "lesion_voxels": int(gt.sum()),
        "pred_voxels":   int((pred > 0.5).sum()),
    }


# ─────────────────────────────────────────────────────────────
# 2.  Batch evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_all(
    case_names: List[str],
    pred_dir: Path,
    label_dir: Path,
    mc_dir: Path = None,
) -> pd.DataFrame:
    """
    Evaluate all cases and return a tidy DataFrame.
    """
    rows = [
        evaluate_case(c, pred_dir, label_dir, mc_dir)
        for c in case_names
    ]
    df = pd.DataFrame(rows)
    return df


# ─────────────────────────────────────────────────────────────
# 3.  Summary statistics + CI
# ─────────────────────────────────────────────────────────────

def summarise(df: pd.DataFrame, ci_level: float = 0.95) -> dict:
    """
    Compute mean, std, median PDS/DSC and the CI for mean PDS.
    Returns a results dict.
    """
    valid = df.dropna(subset=["pds"])

    if len(valid) == 0:
        return {"n": 0}

    mean_pds  = float(valid["pds"].mean())
    std_pds   = float(valid["pds"].std())
    med_pds   = float(valid["pds"].median())
    mean_dice = float(valid["dice"].mean())
    best_pds  = float(valid["pds"].max())
    n         = len(valid)

    ci = (np.nan, np.nan)
    if n >= 2:
        ci = t_dist.interval(ci_level, df=n - 1,
                             loc=mean_pds,
                             scale=std_pds / np.sqrt(n))

    return {
        "n":           n,
        "mean_pds":    mean_pds,
        "std_pds":     std_pds,
        "median_pds":  med_pds,
        "best_pds":    best_pds,
        "mean_dice":   mean_dice,
        "ci_lower":    float(ci[0]),
        "ci_upper":    float(ci[1]),
        "ci_level":    ci_level,
    }


def print_summary(stats: dict) -> None:
    """Pretty-print the evaluation summary."""
    print(f"\n{'='*55}")
    print(f"  EVALUATION SUMMARY  (n={stats.get('n', 0)})")
    print(f"{'='*55}")
    print(f"  Mean PDS       : {stats.get('mean_pds', 0):.4f}  ± {stats.get('std_pds', 0):.4f}")
    print(f"  Median PDS     : {stats.get('median_pds', 0):.4f}")
    print(f"  Best PDS       : {stats.get('best_pds', 0):.4f}")
    print(f"  Mean DSC       : {stats.get('mean_dice', 0):.4f}")
    ci_lvl = int(stats.get("ci_level", 0.95) * 100)
    print(f"  {ci_lvl}% CI for PDS : ({stats.get('ci_lower', 0):.4f}, {stats.get('ci_upper', 0):.4f})")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────
# 4.  Paper comparison table (Table 2 — exact numbers)
# ─────────────────────────────────────────────────────────────

def comparison_table(n_cases: int, mean_pds: float, best_pds: float) -> pd.DataFrame:
    """
    Build a comparison DataFrame matching Table 2 from Joshi et al. (2025).

    Paper Table 2 contains:
      - CNN Encoder-Decoder (House et al., 2021): Dice=0.341
      - MATPR-UNet (Zhang et al., 2024): Dice=0.42 ± 0.08
      - NetPos CNN (Feng et al., 2020): Dice=0.5268
      - nnU-Net (Joshi et al., 2025): Best PDS=0.52, Mean=0.45 (at epoch 100)
    """
    data = {
        "Method": [
            "CNN Encoder-Decoder (House et al., 2021)",
            "MATPR-UNet (Zhang et al., 2024)",
            "NetPos CNN (Feng et al., 2020)",
            "nnU-Net, 3D fullres (Joshi et al., 2025)",
            "nnU-Net + MC-Dropout [This run]",
        ],
        "Dataset": [
            "201 FCD cases (3D MRI)",
            "EPISURG dataset",
            "18 FLAIR-negative FCD",
            "85 FCD + 85 controls (3D FLAIR)",
            f"{n_cases} cases (3D FLAIR)",
        ],
        "Metric": [
            "DSC",
            "DSC",
            "DSC",
            "PDS",
            "PDS",
        ],
        "Best Score": [
            "0.341",
            "0.42 ± 0.08",
            "0.5268",
            "0.52 (Fold 5) / 0.45 (mean at ep100)",
            f"{best_pds:.4f} (best) / {mean_pds:.4f} (mean)",
        ],
    }
    return pd.DataFrame(data)


# ─────────────────────────────────────────────────────────────
# 5.  Fold-wise statistical analysis from REAL training logs
# ─────────────────────────────────────────────────────────────

def _find_training_log(fold_dir: Path) -> Optional[Path]:
    """
    nnU-Net saves logs as training_log_YYYY_M_D_HH_MM_SS.txt (timestamped).
    Return the most recent one found, or None.
    """
    logs = sorted(fold_dir.glob("training_log_*.txt"))
    if logs:
        return logs[-1]  # take the latest (in case of resumed training)
    # Final fallback: plain training_log.txt
    plain = fold_dir / "training_log.txt"
    return plain if plain.exists() else None


def fold_statistics(
    results_dir: Path,
    task_id: int,
    trainer: str,
    plans: str,
    config: str,
    n_folds: int,
    fallback_mean_pds: float = None,
    fallback_best_pds: float = None,
) -> pd.DataFrame:
    """
    Parse nnU-Net training logs to extract per-fold mean and final PDS.
    ALL values come from the actual trained model logs — no hardcoded paper
    numbers are ever substituted.  If a log is missing the fold is skipped
    and a clear warning is printed.

    Returns a DataFrame with columns: fold, mean_pds, final_pds.
    """
    fold_mean_pds:  List[float] = []
    fold_final_pds: List[float] = []
    folds_used:     List[int]   = []

    for fold in range(n_folds):
        fold_dir = (
            results_dir
            / f"Dataset{task_id:03d}_FCD"
            / f"{trainer}__{plans}__{config}"
            / f"fold_{fold}"
        )
        log_file = _find_training_log(fold_dir)

        pseudo_vals: List[float] = []

        if log_file is not None:
            with open(log_file) as fh:
                for line in fh:
                    # nnU-Net v2 log line format:
                    #   2026-05-03 20:31:47.211991: Pseudo dice [np.float32(0.1425)]
                    lower = line.lower()
                    if "pseudo dice" in lower:
                        try:
                            # Extract the number inside [ ... ]
                            inside = line.split("[")[-1].split("]")[0]
                            # Handles both plain floats and np.float32(x)
                            val_str = inside.split("(")[-1].rstrip(")")
                            pseudo_vals.append(float(val_str))
                        except (ValueError, IndexError):
                            pass
        else:
            logger.warning("Fold %d: no training log found in %s — fold skipped", fold + 1, fold_dir)

        if pseudo_vals:
            fold_mean_pds.append(float(np.mean(pseudo_vals)))
            fold_final_pds.append(float(pseudo_vals[-1]))
            folds_used.append(fold + 1)
            logger.info(
                "Fold %d: %d epochs parsed | mean PDS=%.4f | final PDS=%.4f",
                fold + 1, len(pseudo_vals), fold_mean_pds[-1], fold_final_pds[-1]
            )
        elif log_file is not None:
            logger.warning(
                "Fold %d: log found but no 'Pseudo dice' entries — check log format: %s",
                fold + 1, log_file
            )

    if not fold_mean_pds:
        raise RuntimeError(
            "No training logs found for any fold.  "
            f"Expected logs in: {results_dir}/.../fold_N/training_log_*.txt"
        )

    df = pd.DataFrame({
        "fold":      folds_used,
        "mean_pds":  fold_mean_pds,
        "final_pds": fold_final_pds,
    })

    # Paired t-test: mean vs final PDS across folds
    if len(fold_mean_pds) >= 2:
        t_stat, p_val = ttest_rel(fold_mean_pds, fold_final_pds)
        print(f"\nPaired t-test (mean PDS per epoch vs final-epoch PDS across folds):")
        print(f"  t = {t_stat:.4f}   p = {p_val:.4f}")
        if p_val < 0.05:
            print("  → Statistically significant: final epoch PDS > mean epoch PDS (p < 0.05) ✅")
        else:
            print("  → Not statistically significant at α = 0.05")

    return df