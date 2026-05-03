"""
inference.py — nnU-Net inference + MC-Dropout uncertainty estimation.

Two inference modes:
  A. Standard ensemble inference via nnUNetv2_predict CLI (all 5 folds)
  B. MC-Dropout inference: T stochastic forward passes with dropout ACTIVE
     during inference → voxel-wise mean probability + predictive entropy

Paper (Joshi et al., 2025):
  - Input: single-channel FLAIR (top-5 axial slices)
  - Standard: 5-fold ensemble using checkpoint_best.pth
  - MC-Dropout: T=20 forward passes, p=0.2 dropout
"""

import numpy as np
import nibabel as nib
import subprocess
import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple
from scipy import ndimage

logger = logging.getLogger("fcd.inference")


# ─────────────────────────────────────────────────────────────
# 1.  Standard nnU-Net ensemble inference (CLI wrapper)
# ─────────────────────────────────────────────────────────────

def run_nnunet_inference(
    input_dir: Path,
    output_dir: Path,
    task_id: int,
    config: str = "3d_fullres",
    folds: str = "all",
    save_probabilities: bool = True,
    checkpoint_name: str = "checkpoint_best.pth",
    trainer: str = "nnUNetTrainer_100epochs",
    plans: str = "nnUNetPlans",
) -> bool:
    """
    Run nnUNetv2_predict on all cases in input_dir.

    Paper: ensemble prediction using all 5 fold checkpoints.

    Args:
        input_dir          : directory containing *_0000.nii.gz images (FLAIR only)
        output_dir         : where predictions are saved
        task_id            : nnU-Net task integer ID
        config             : '3d_fullres' (paper setting)
        folds              : 'all' or comma-separated fold numbers
        save_probabilities : if True, also saves softmax .pkl files
        checkpoint_name    : name of the checkpoint to use

    Returns True on success.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    import shutil
    import sys
    import os
    python_dir = os.path.dirname(sys.executable)
    cmd_bin    = os.path.join(python_dir, "nnUNetv2_predict")

    if not os.path.exists(cmd_bin):
        cmd_bin = shutil.which("nnUNetv2_predict") or "nnUNetv2_predict"

    cmd = [
        cmd_bin,
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", str(task_id),
        "-c", config,
        "-f", *folds.split(","),
        "-chk", checkpoint_name,
        "-tr", trainer,
        "-p", plans,
    ]
    if save_probabilities:
        cmd.append("--save_probabilities")

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd)

    if result.returncode == 0:
        logger.info("✅  Inference complete")
        n_preds = len(list(output_dir.glob("*.nii.gz")))
        logger.info("   %d prediction files generated", n_preds)
        return True
    else:
        logger.warning("⚠️  Inference returned code %d", result.returncode)
        return False


# ─────────────────────────────────────────────────────────────
# 2.  MC-Dropout inference
# ─────────────────────────────────────────────────────────────

def _enable_dropout(model: nn.Module) -> None:
    """Set all dropout layers to train mode (so they are active)."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


def _inject_dropout(model: nn.Module, p: float = 0.2) -> None:
    """Insert Dropout3d after every Conv3d / ConvTranspose3d layer."""
    for name, module in list(model.named_children()):
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            if not (isinstance(module, nn.Sequential) and any(isinstance(c, nn.Dropout3d) for c in module)):
                setattr(model, name, nn.Sequential(module, nn.Dropout3d(p=p)))
        else:
            _inject_dropout(module, p)


def load_nnunet_predictor(
    results_dir: Path,
    task_id: int,
    trainer: str,
    plans: str,
    config: str,
    fold: int = 0,
    checkpoint_name: str = "checkpoint_best.pth",
    device: str = "cuda",
):
    """
    Load an nnUNetPredictor from a trained model folder.
    Returns the predictor object (has .network attribute).
    """
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=(device == "cuda"),
        device=torch.device(device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    model_folder = (
        results_dir
        / f"Dataset{task_id:03d}_FCD"
        / f"{trainer}__{plans}__{config}"
    )

    if not model_folder.exists():
        raise FileNotFoundError(f"Model folder not found: {model_folder}")

    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=(fold,),
        checkpoint_name=checkpoint_name,
    )

    logger.info("✅  Model loaded from %s", model_folder)
    return predictor


@torch.no_grad()
def mc_dropout_predict(
    predictor,
    flair: np.ndarray,
    n_samples: int = 20,
    dropout_p: float = 0.2,
    confidence_thresh: float = 0.5,
    device: str = "cuda",
) -> dict:
    """
    Run T stochastic forward passes with dropout active.

    Paper: T=20 samples, p=0.2 dropout probability.
    Input: single-channel FLAIR (K, H, W) where K=top-5 slices.

    Args:
        predictor         : loaded nnUNetPredictor
        flair             : (K, H, W) float32 preprocessed FLAIR (top-5 slices)
        n_samples         : T — number of MC samples (paper: 20)
        dropout_p         : dropout probability (paper: 0.2)
        confidence_thresh : threshold applied to mean probability

    Returns dict with:
        mean_prob   : (K, H, W) float32
        uncertainty : (K, H, W) float32  — predictive entropy
        hard_pred   : (K, H, W) uint8    — binary mask
    """
    if not hasattr(predictor, "network"):
        logger.warning("predictor.network not found; skipping MC-Dropout")
        return {}

    # Inject and enable dropout
    _inject_dropout(predictor.network, p=dropout_p)
    predictor.network.to(device)

    # nnU-Net expects (Batch, Channel, X, Y, Z)
    # Our FLAIR is (K, H, W) → single channel → (1, 1, H, W, K)
    flair_t = np.transpose(flair, (1, 2, 0))  # (H, W, K)

    x = torch.from_numpy(
        flair_t[np.newaxis, np.newaxis]  # (1, 1, H, W, K)
    ).float().to(device)

    probs = []
    for _ in range(n_samples):
        predictor.network.eval()
        _enable_dropout(predictor.network)

        logits = predictor.network(x)
        # Handle cases where network returns a list (deep supervision)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]

        prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()[0]  # (H, W, K)
        probs.append(prob)

    probs_arr = np.stack(probs, axis=0)       # (T, H, W, K)
    mean_prob = probs_arr.mean(axis=0)        # (H, W, K)

    eps       = 1e-8
    p         = np.clip(mean_prob, eps, 1 - eps)
    entropy   = -(p * np.log(p) + (1 - p) * np.log(1 - p))

    hard_pred = (mean_prob > confidence_thresh).astype(np.uint8)

    # Transpose back to (K, H, W)
    mean_prob = np.transpose(mean_prob, (2, 0, 1))
    entropy   = np.transpose(entropy, (2, 0, 1))
    hard_pred = np.transpose(hard_pred, (2, 0, 1))

    return {
        "mean_prob":   mean_prob.astype(np.float32),
        "uncertainty": entropy.astype(np.float32),
        "hard_pred":   hard_pred,
    }


# ─────────────────────────────────────────────────────────────
# 3.  Post-processing
# ─────────────────────────────────────────────────────────────

def postprocess_prediction(
    hard_pred: np.ndarray,
    mean_prob: np.ndarray,
    uncertainty: np.ndarray,
    confidence_thresh: float = 0.5,
    uncertainty_thresh: float = 0.3,
    keep_largest_cc: bool = True,
) -> np.ndarray:
    """
    Refine a binary prediction mask:
      1. Remove high-uncertainty voxels
      2. Remove low-confidence voxels
      3. Keep only the largest connected component (optional)

    Returns refined uint8 binary mask.
    """
    refined = hard_pred.copy()
    refined[uncertainty > uncertainty_thresh] = 0
    refined[mean_prob   < confidence_thresh ] = 0

    if keep_largest_cc and refined.sum() > 0:
        labeled, n_comps = ndimage.label(refined)
        if n_comps > 1:
            sizes       = ndimage.sum(refined, labeled, range(1, n_comps + 1))
            largest_idx = int(np.argmax(sizes)) + 1
            refined     = (labeled == largest_idx).astype(np.uint8)

    return refined.astype(np.uint8)


# ─────────────────────────────────────────────────────────────
# 4.  Full per-case MC-Dropout inference pipeline
# ─────────────────────────────────────────────────────────────

def run_mc_inference_for_case(
    case_name: str,
    task_dir: Path,
    mc_out_dir: Path,
    predictor,
    mc_cfg: dict,
    device: str = "cuda",
) -> dict:
    """
    End-to-end MC-Dropout inference for one case.
    Single-channel FLAIR input (top-5 slices).
    Saves prediction, probability map, and uncertainty map as NIfTI.
    Computes Dice / PDS if ground truth is available.

    Returns result dict.
    """
    from src.utils import pseudo_dice_score, dice_score

    # Single-channel FLAIR only (_0000)
    flair_path = task_dir / "imagesTr" / f"{case_name}_0000.nii.gz"
    lbl_path   = task_dir / "labelsTr" / f"{case_name}.nii.gz"

    if not flair_path.exists():
        return {"case": case_name, "status": "missing_input"}

    flair = nib.load(str(flair_path)).get_fdata().astype(np.float32)

    mc_out = mc_dropout_predict(
        predictor,
        flair,
        n_samples         = mc_cfg["n_samples"],
        dropout_p         = mc_cfg["dropout_p"],
        confidence_thresh = mc_cfg["confidence_thresh"],
        device            = device,
    )

    if not mc_out:
        return {"case": case_name, "status": "no_network"}

    refined = postprocess_prediction(
        mc_out["hard_pred"],
        mc_out["mean_prob"],
        mc_out["uncertainty"],
        confidence_thresh  = mc_cfg["confidence_thresh"],
        uncertainty_thresh = mc_cfg["uncertainty_thresh"],
    )

    # Save outputs
    mc_out_dir.mkdir(parents=True, exist_ok=True)
    ref_nib = nib.load(str(flair_path))
    affine  = ref_nib.affine

    nib.save(nib.Nifti1Image(refined.astype(np.uint8),          affine),
             str(mc_out_dir / f"{case_name}_pred.nii.gz"))
    nib.save(nib.Nifti1Image(mc_out["mean_prob"],                affine),
             str(mc_out_dir / f"{case_name}_prob.nii.gz"))
    nib.save(nib.Nifti1Image(mc_out["uncertainty"],              affine),
             str(mc_out_dir / f"{case_name}_uncertainty.nii.gz"))

    result = {
        "case":            case_name,
        "status":          "ok",
        "mean_prob_max":   float(mc_out["mean_prob"].max()),
        "mean_entropy":    float(mc_out["uncertainty"].mean()),
    }

    if lbl_path.exists():
        gt              = nib.load(str(lbl_path)).get_fdata().astype(np.uint8)
        result["pds"]   = pseudo_dice_score(refined, gt)
        result["dice"]  = dice_score(refined, gt)

    return result