"""
preprocessing.py — MRI preprocessing pipeline (Joshi et al., 2025).

Steps per subject — EXACTLY matching the paper (Section 3.2 & 3.3):
  1. Load 3D FLAIR NIfTI
  2. Resample to 1 mm isotropic
  3. Skull-strip FLAIR via HD-BET (or Otsu fallback)
  4. Z-score intensity normalisation within brain mask
  5. Select top-5 axial slices by:
       - FCD patients  : slices with MOST lesion/ROI voxels (label-guided)
       - Healthy controls: slices with HIGHEST peak FLAIR intensity
  6. Save SINGLE-CHANNEL NIfTI (FLAIR only) for nnU-Net
     → output shape: (5, H, W) per subject

NOTE: The paper uses ONLY FLAIR as input (single channel).
      T1 is NOT used as a model input (Joshi et al., 2025, Section 3.3).
"""

import gc
import warnings
import subprocess
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from scipy.ndimage import zoom
from typing import Optional, Tuple

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# 1.  Resampling
# ─────────────────────────────────────────────────────────────

def resample_to_spacing(
    sitk_image: sitk.Image,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    interpolator=sitk.sitkLinear,
) -> sitk.Image:
    """
    Resample a SimpleITK image to target voxel spacing (mm).
    Standardises all scans to 1 mm isotropic resolution (paper Section 3.2).
    """
    original_spacing = sitk_image.GetSpacing()
    original_size    = sitk_image.GetSize()

    new_size = [
        int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(interpolator)
    return resampler.Execute(sitk_image)


# ─────────────────────────────────────────────────────────────
# 2.  Skull stripping
# ─────────────────────────────────────────────────────────────

def skull_strip(
    image_path: Path,
    out_dir: Path,
    use_hdbet: bool = True,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Skull-strip a NIfTI image.
    Primary method: HD-BET (deep learning, CPU or GPU).
    Fallback:       Otsu threshold masking (SimpleITK).

    Returns:
        stripped_array  : (Z,H,W) float32 — brain only
        brain_mask      : (Z,H,W) uint8   — binary brain mask
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stem     = image_path.name.replace(".nii.gz", "").replace(".nii", "")
    out_path = out_dir / f"{stem}_bet.nii.gz"

    if use_hdbet:
        try:
            if out_path.exists():
                print("    HD-BET output cached. Skipping inference.")
                data = nib.load(str(out_path)).get_fdata().astype(np.float32)
                mask = (data > 0).astype(np.uint8)
                return data, mask

            import shutil
            import sys
            import os
            python_dir = os.path.dirname(sys.executable)
            hdbet_bin  = os.path.join(python_dir, "hd-bet")

            if not os.path.exists(hdbet_bin):
                hdbet_bin = shutil.which("hd-bet") or "hd-bet"

            hdbet_device = "cuda" if device == "cuda" else "cpu"

            result = subprocess.run(
                [
                    hdbet_bin, "-i", str(image_path), "-o", str(out_path),
                    "-device", hdbet_device, "--disable_tta",
                ],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode == 0 and out_path.exists():
                data = nib.load(str(out_path)).get_fdata().astype(np.float32)
                mask = (data > 0).astype(np.uint8)
                return data, mask
            else:
                print(f"    HD-BET returned {result.returncode}; using Otsu fallback")
        except Exception as exc:
            print(f"    HD-BET exception ({exc}); using Otsu fallback")


    # ── Otsu fallback ─────────────────────────────────────────
    img  = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
    arr  = sitk.GetArrayFromImage(img).astype(np.float32)

    otsu = sitk.OtsuThresholdImageFilter()
    otsu.SetInsideValue(0)
    otsu.SetOutsideValue(1)
    mask_img = otsu.Execute(img)
    mask     = sitk.GetArrayFromImage(mask_img).astype(np.uint8)

    return (arr * mask).astype(np.float32), mask


# ─────────────────────────────────────────────────────────────
# 3.  Intensity normalisation
# ─────────────────────────────────────────────────────────────

def normalize_intensity(
    volume: np.ndarray,
    mask: np.ndarray,
    clip_range: Tuple[float, float] = (-5.0, 5.0),
) -> np.ndarray:
    """
    Z-score normalise voxel intensities within the brain mask.
    Background (mask == 0) is set to zero.
    Values are clipped to clip_range to suppress outliers.
    """
    brain_voxels = volume[mask > 0]
    if brain_voxels.size == 0:
        return volume.astype(np.float32)

    mu    = float(brain_voxels.mean())
    sigma = float(brain_voxels.std()) + 1e-8
    normed = (volume - mu) / sigma
    normed[mask == 0] = 0.0
    return np.clip(normed, clip_range[0], clip_range[1]).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# 4.  Top-K slice selection — paper's key preprocessing novelty
#     (Joshi et al., 2025, Section 3.3)
# ─────────────────────────────────────────────────────────────

def select_top_k_slices(
    flair_volume: np.ndarray,
    label_volume: Optional[np.ndarray] = None,
    k: int = 5,
) -> Tuple[np.ndarray, Optional[np.ndarray], list]:
    """
    Select top-K axial slices as described in the paper (Section 3.3):

    - FCD patients (label_volume provided & non-zero):
        Select the K slices containing the MOST lesion/ROI voxels.
        This ensures the model is trained on the slices most relevant
        to the FCD lesion.

    - Healthy controls (no label or all-zero label):
        Select the K slices with the HIGHEST peak FLAIR intensity.
        This mimics how an FCD lesion would appear (hyperintense on FLAIR).

    Returns:
        selected_flair  : (K, H, W) float32
        selected_labels : (K, H, W) uint8 or None
        top_idx         : list of K selected axial slice indices (sorted)
    """
    n_slices = flair_volume.shape[0]

    if label_volume is not None and label_volume.sum() > 0:
        # FCD patient: pick slices with most lesion voxels (label-guided)
        slice_sums = label_volume.sum(axis=(1, 2))   # shape (Z,)
        top_idx    = sorted(np.argsort(slice_sums)[::-1][:k].tolist())
    else:
        # Healthy control or no lesion: pick brightest FLAIR slices
        peak_int = np.array([flair_volume[z].max() for z in range(n_slices)])
        top_idx  = sorted(np.argsort(peak_int)[::-1][:k].tolist())

    selected_flair  = flair_volume[top_idx]                              # (K, H, W)
    selected_labels = label_volume[top_idx] if label_volume is not None else None

    return selected_flair, selected_labels, top_idx


# ─────────────────────────────────────────────────────────────
# 5.  Full per-subject preprocessing pipeline
#     Matches paper Figure 3 / Section 3.2–3.3 exactly
# ─────────────────────────────────────────────────────────────

def preprocess_subject(
    entry: dict,
    task_dir: Path,
    skull_dir: Path,
    case_id: int,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    top_k_slices: int = 5,
    use_skull_strip: bool = True,
    device: str = "cuda",
) -> dict:
    """
    Complete preprocessing pipeline for one subject.

    Pipeline (matches Joshi et al., 2025):
      1. Load + resample FLAIR to 1 mm isotropic
      2. Skull-strip (HD-BET or Otsu fallback)
      3. Z-score intensity normalisation
      4. Load + resample label (FCD patients only)
      5. Select top-K axial slices
         - FCD: K slices with most lesion voxels
         - Controls: K slices with highest FLAIR peak intensity
      6. Save SINGLE-CHANNEL FLAIR NIfTI (shape: K×H×W) for nnU-Net
         (T1 is NOT used as model input — paper uses FLAIR only)

    Args:
        entry   : dict with keys 'subject_id', 'flair', 't1', 'label'
        task_dir: nnU-Net task root (imagesTr / labelsTr will be created here)
        skull_dir: directory for HD-BET outputs
        case_id : integer counter for naming (FCD_0001, etc.)

    Returns:
        dict with status, case_name, shape, lesion_voxels, sel_idx, etc.
    """
    sid    = entry["subject_id"]
    result = {"subject_id": sid, "case_id": case_id, "status": "failed"}

    try:
        # ── Load + resample FLAIR ─────────────────────────────
        flair_sitk = sitk.ReadImage(str(entry["flair"]), sitk.sitkFloat32)
        flair_sitk = resample_to_spacing(flair_sitk, target_spacing)
        flair_np   = sitk.GetArrayFromImage(flair_sitk).astype(np.float32)

        # ── Skull stripping on FLAIR ──────────────────────────
        sub_skull_dir = skull_dir / sid
        flair_stripped, brain_mask = skull_strip(
            entry["flair"], sub_skull_dir,
            use_hdbet=use_skull_strip, device=device,
        )

        # Resize skull-stripped to match resampled FLAIR
        if flair_stripped.shape != flair_np.shape:
            zf             = [flair_np.shape[i] / flair_stripped.shape[i] for i in range(3)]
            flair_stripped = zoom(flair_stripped, zf, order=1).astype(np.float32)
            brain_mask     = zoom(brain_mask,     zf, order=0).astype(np.uint8)

        flair_np = flair_np * (brain_mask > 0)

        # ── Intensity normalisation ───────────────────────────
        flair_norm = normalize_intensity(flair_np, brain_mask)

        # ── Load + resample label ─────────────────────────────
        label_np = None
        if entry.get("label") is not None and Path(str(entry["label"])).exists():
            lbl_sitk = sitk.ReadImage(str(entry["label"]), sitk.sitkUInt8)
            lbl_sitk = resample_to_spacing(lbl_sitk, target_spacing, sitk.sitkNearestNeighbor)
            label_np = sitk.GetArrayFromImage(lbl_sitk).astype(np.uint8)
            if label_np.shape != flair_norm.shape:
                zf       = [flair_norm.shape[i] / label_np.shape[i] for i in range(3)]
                label_np = zoom(label_np, zf, order=0).astype(np.uint8)

        # ── Top-K slice selection (paper's key novelty) ───────
        # FCD patients: select slices with most lesion voxels
        # Controls:     select slices with highest FLAIR intensity
        flair_sel, label_sel, sel_idx = select_top_k_slices(
            flair_volume=flair_norm,
            label_volume=label_np,
            k=top_k_slices,
        )

        # For controls without labels: create all-zero label volume
        if label_sel is None:
            label_sel = np.zeros(flair_sel.shape, dtype=np.uint8)

        # ── Save SINGLE-CHANNEL FLAIR NIfTI for nnU-Net ───────
        # Paper: input is FLAIR only (single channel _0000)
        # Output shape: (K, H, W) where K = top_k_slices (default 5)
        case_name = f"FCD_{case_id:04d}"
        img_dir   = task_dir / "imagesTr"
        lbl_dir   = task_dir / "labelsTr"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        # Build affine from resampled FLAIR (preserve physical metadata)
        spacing   = flair_sitk.GetSpacing()
        origin    = flair_sitk.GetOrigin()
        direction = np.array(flair_sitk.GetDirection()).reshape(3, 3)
        affine    = np.eye(4)
        affine[:3, :3] = direction * np.array(spacing)
        affine[:3, 3]  = origin

        # Single-channel: _0000 = FLAIR (paper does NOT use T1 as input)
        nib.save(
            nib.Nifti1Image(flair_sel, affine),
            str(img_dir / f"{case_name}_0000.nii.gz"),
        )
        nib.save(
            nib.Nifti1Image(label_sel.astype(np.uint8), affine),
            str(lbl_dir / f"{case_name}.nii.gz"),
        )

        result.update({
            "status":        "ok",
            "case_name":     case_name,
            "flair_shape":   tuple(flair_sel.shape),
            "n_slices":      flair_sel.shape[0],
            "lesion_voxels": int(label_sel.sum()),
            "sel_idx":       sel_idx,
        })

    except Exception as exc:
        result["error"] = str(exc)
        print(f"  [ERROR] {sid}: {exc}")

    gc.collect()
    return result