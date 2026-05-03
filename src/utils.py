"""
utils.py — Shared utility functions for the FCD segmentation pipeline.
Covers: config loading, metric functions, logging helpers.
"""

import yaml
import logging
import numpy as np
from pathlib import Path
from typing import Union, Optional

# ─────────────────────────────────────────────────────────────
# 1.  Config loader
# ─────────────────────────────────────────────────────────────

def load_config(config_path: Union[str, Path] = None) -> dict:
    """
    Load the YAML configuration file.
    Resolves all relative paths to absolute paths anchored at the project root.
    """
    if config_path is None:
        # Default: configs/config.yaml relative to this file's parent-parent
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve paths relative to project root
    root = config_path.parent.parent
    cfg["_root"]            = root
    cfg["_nnunet_raw"]      = root / cfg["dataset"]["work_dir"] / "nnUNet_raw"
    cfg["_nnunet_prep"]     = root / cfg["dataset"]["work_dir"] / "nnUNet_preprocessed"
    cfg["_nnunet_results"]  = root / cfg["dataset"]["work_dir"] / "nnUNet_results"
    cfg["_dataset_dir"]     = root / cfg["dataset"]["base_dir"]
    cfg["_skull_dir"]       = root / cfg["dataset"]["work_dir"] / "skull_stripped"
    cfg["_pred_dir"]        = root / cfg["dataset"]["work_dir"] / "predictions"
    cfg["_mc_dir"]          = root / cfg["dataset"]["work_dir"] / "mc_dropout_predictions"
    cfg["_figures_dir"]     = root / cfg["output"]["figures_dir"]

    task_id   = cfg["nnunet"]["task_id"]
    task_name = f"Dataset{task_id:03d}_FCD"
    cfg["_task_name"] = task_name
    cfg["_task_dir"]  = cfg["_nnunet_raw"] / task_name

    return cfg


# ─────────────────────────────────────────────────────────────
# 2.  Logging setup
# ─────────────────────────────────────────────────────────────

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure a named logger with timestamp formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ─────────────────────────────────────────────────────────────
# 3.  Metric functions
# ─────────────────────────────────────────────────────────────

def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Standard Dice Similarity Coefficient:  DSC = 2|A∩B| / (|A|+|B|)
    Returns 1.0 if both arrays are empty (trivially perfect).
    """
    pred = (pred > 0.5).astype(np.float32)
    gt   = (gt > 0.5).astype(np.float32)
    intersection = (pred * gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * intersection / denom)


def pseudo_dice_score(pred: np.ndarray,
                      gt: np.ndarray,
                      eps: float = 1e-6) -> float:
    """
    Stabilised Pseudo-Dice Score (as used in the paper):
        PDS = (2 * |A∩B| + ε) / (|A| + |B| + ε)
    Handles small/empty lesion volumes gracefully.
    """
    pred = (pred > 0.5).astype(np.float32)
    gt   = (gt > 0.5).astype(np.float32)
    intersection = (pred * gt).sum()
    return float((2.0 * intersection + eps) / (pred.sum() + gt.sum() + eps))


def sensitivity(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    True positive rate = TP / (TP + FN).
    Returns 1.0 if the ground truth is empty (no positive samples).
    """
    pred = (pred > 0.5).astype(np.float32)
    gt   = (gt > 0.5).astype(np.float32)
    tp = (pred * gt).sum()
    fn = gt.sum() - tp
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 1.0


def specificity(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    True negative rate = TN / (TN + FP).
    Returns 1.0 if the ground truth is all-positive (no negative samples).
    """
    pred = (pred > 0.5).astype(np.float32)
    gt   = (gt > 0.5).astype(np.float32)
    tn = ((1 - pred) * (1 - gt)).sum()
    fp = pred.sum() - (pred * gt).sum()
    denom = tn + fp
    return float(tn / denom) if denom > 0 else 1.0


def hausdorff_distance_95(pred: np.ndarray, gt: np.ndarray,
                           voxel_spacing: float = 1.0) -> float:
    """
    95th percentile Hausdorff distance (HD95) in mm.
    Falls back to 0.0 if either array is empty.
    """
    from scipy.ndimage import binary_erosion

    pred_bin = (pred > 0.5)
    gt_bin   = gt.astype(bool)

    if not pred_bin.any() or not gt_bin.any():
        return 0.0

    # Surface voxels
    pred_surf = pred_bin & ~binary_erosion(pred_bin)
    gt_surf   = gt_bin   & ~binary_erosion(gt_bin)

    pred_pts = np.argwhere(pred_surf).astype(float) * voxel_spacing
    gt_pts   = np.argwhere(gt_surf).astype(float)   * voxel_spacing

    # Bidirectional max distance at 95th percentile
    from scipy.spatial import cKDTree
    tree_gt   = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)

    d_pred_to_gt, _ = tree_gt.query(pred_pts)
    d_gt_to_pred, _ = tree_pred.query(gt_pts)

    hd95 = max(np.percentile(d_pred_to_gt, 95),
               np.percentile(d_gt_to_pred, 95))
    return float(hd95)


# ─────────────────────────────────────────────────────────────
# 4.  Directory helpers
# ─────────────────────────────────────────────────────────────

def ensure_dirs(*dirs) -> None:
    """Create all given directories (including parents) if they don't exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def set_nnunet_env(cfg: dict) -> None:
    """Set the three environment variables required by nnU-Net."""
    import os
    os.environ["nnUNet_raw"]          = str(cfg["_nnunet_raw"])
    os.environ["nnUNet_preprocessed"] = str(cfg["_nnunet_prep"])
    os.environ["nnUNet_results"]      = str(cfg["_nnunet_results"])