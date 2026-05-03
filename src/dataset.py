"""
dataset.py — Subject discovery and nnU-Net dataset JSON builder.

Handles the OpenNeuro ds004199 BIDS layout:
  sub-XXX/
    anat/           OR    ses-preop/anat/
      *FLAIR*.nii.gz
      *T1w*.nii.gz
      *roi*.nii.gz   ← lesion label (may also be in derivatives/)

Paper (Joshi et al., 2025) specifics:
  - 85 FCD patients + 85 healthy controls (170 total)
  - Both FCD and controls are used in training
  - FCD: labelled with ROI masks; Controls: all-zero labels
  - Single-channel FLAIR input (NOT T1)
"""

import json
import random
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────────────────────
# 1.  Subject discovery
# ─────────────────────────────────────────────────────────────

def discover_subjects(
    dataset_dir: Path,
    use_subset: bool = True,
    n_subset: int = 10,
) -> Tuple[List[dict], List[dict]]:
    """
    Parse participants.tsv and locate MRI files for every subject.

    Returns:
        fcd_subjects     : list of dicts for FCD patients
        control_subjects : list of dicts for healthy controls
    """
    dataset_dir = Path(dataset_dir)
    participants_file = dataset_dir / "participants.tsv"

    if not participants_file.exists():
        raise FileNotFoundError(f"participants.tsv not found at {participants_file}")

    df = pd.read_csv(participants_file, sep="\t")
    print(f"\n📄  Participants loaded — total entries: {len(df)}")
    print(df[["participant_id", "group"]].head(5).to_string(index=False))

    fcd_subjects: List[dict]     = []
    control_subjects: List[dict] = []

    print("\n🔍  Scanning subjects ...\n")

    for _, row in df.iterrows():
        sid   = row["participant_id"]
        group = str(row.get("group", "unknown")).lower().strip()

        sub_dir = dataset_dir / sid
        if not sub_dir.exists():
            print(f"  ⚠️  Missing directory: {sid}")
            continue

        # ── Locate anat folder (handles ses-preop layout) ─────
        anat = sub_dir / "anat"
        if not anat.exists():
            anat = sub_dir / "ses-preop" / "anat"
        if not anat.exists():
            print(f"  ⚠️  No anat folder: {sid}")
            continue

        # ── Find FLAIR file (primary modality) ────────────────
        flair_files = list(anat.glob("*FLAIR*.nii*"))

        if not flair_files:
            print(f"  ❌  No FLAIR: {sid}")
            continue

        flair_path = flair_files[0]

        # ── Find T1 (for registration reference only — NOT model input) ──
        t1_files = list(anat.glob("*T1w*.nii*"))
        t1_path  = t1_files[0] if t1_files else None

        # ── Find lesion label (ROI) ───────────────────────────
        label_path = _find_label(anat, dataset_dir, sid)

        entry = {
            "subject_id": sid,
            "flair":      flair_path,
            "t1":         t1_path,    # kept for reference, not used as model input
            "label":      label_path,
        }

        has_label = "YES ✅" if label_path else "NO  ❌"
        print(f"  {sid}  group={group:<8}  label={has_label}")

        if group == "fcd":
            fcd_subjects.append(entry)
        else:
            control_subjects.append(entry)

    # ── Apply subset if requested ─────────────────────────────
    if use_subset:
        fcd_subjects     = fcd_subjects[:n_subset]
        control_subjects = control_subjects[:n_subset]

    print(f"\n{'='*50}")
    print(f"  FCD subjects     : {len(fcd_subjects)}")
    print(f"  Control subjects : {len(control_subjects)}")
    print(f"{'='*50}\n")

    if fcd_subjects:
        s = fcd_subjects[0]
        print("📌  Example FCD subject:")
        print(f"    ID    : {s['subject_id']}")
        print(f"    FLAIR : {s['flair']}")
        print(f"    T1    : {s['t1']}")
        print(f"    Label : {s['label']}")

    return fcd_subjects, control_subjects


def _find_label(
    anat_dir: Path,
    dataset_dir: Path,
    sid: str,
) -> Optional[Path]:
    """
    Look for a lesion ROI label in two locations:
    1. Inside the anat/ folder (primary, ds004199 convention)
    2. Inside derivatives/ (fallback)
    """
    # Primary: *roi*.nii* inside anat
    roi_files = list(anat_dir.glob("*roi*.nii*"))
    if roi_files:
        return roi_files[0]

    # Fallback: derivatives
    deriv = dataset_dir / "derivatives"
    if deriv.exists():
        found = list(deriv.rglob(f"{sid}*roi*.nii*"))
        if found:
            return found[0]

    return None


# ─────────────────────────────────────────────────────────────
# 2.  Filter to labelled subjects only (FCD patients)
# ─────────────────────────────────────────────────────────────

def filter_labeled(subjects: List[dict]) -> Tuple[List[dict], int]:
    """
    Keep only subjects that have a ground-truth lesion label.
    Returns (valid_list, n_skipped).
    """
    valid   = [s for s in subjects if s.get("label") is not None]
    skipped = len(subjects) - len(valid)
    return valid, skipped


# ─────────────────────────────────────────────────────────────
# 3.  nnU-Net dataset.json writer
#     Paper: SINGLE-channel FLAIR input only
# ─────────────────────────────────────────────────────────────

def write_dataset_json(
    task_dir: Path,
    case_names: List[str],
    task_name: str,
    seed: int = 42,
) -> dict:
    """
    Write the nnU-Net v2 dataset.json file into task_dir.

    Paper uses FLAIR-only (single channel).
    Controls are included with all-zero labels.

    Returns the dict that was written.
    """
    dataset_json = {
        # Single channel: FLAIR only (matches Joshi et al., 2025)
        "channel_names": {
            "0": "FLAIR",
        },
        "labels": {
            "background": 0,
            "FCD_lesion": 1,
        },
        "numTraining": len(case_names),
        "file_ending": ".nii.gz",
        "name": task_name,
        "description": (
            "FCD Type II lesion segmentation from 3D FLAIR MRI. "
            "Single-channel input. "
            "Based on Joshi et al., Frontiers in AI, 2025."
        ),
        "reference": "Joshi et al., Frontiers in AI, 2025",
        "tensorImageSize": "3D",
        "training": [
            {
                "image": f"./imagesTr/{name}",
                "label": f"./labelsTr/{name}",
            }
            for name in case_names
        ],
    }

    json_path = task_dir / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"✅  dataset.json written → {json_path}")
    return dataset_json


# ─────────────────────────────────────────────────────────────
# 4.  Dataset integrity check
# ─────────────────────────────────────────────────────────────

def check_dataset_integrity(task_dir: Path) -> bool:
    """
    Verify that every imagesTr case has a matching labelsTr file.
    For single-channel FLAIR: check only _0000.nii.gz files.
    Raises RuntimeError on mismatch; returns True on success.
    """
    img_cases_0000 = {
        f.name.replace("_0000.nii.gz", "")
        for f in (task_dir / "imagesTr").glob("*_0000.nii.gz")
    }
    lbl_cases = {
        f.name.replace(".nii.gz", "")
        for f in (task_dir / "labelsTr").glob("*.nii.gz")
    }

    missing_labels = img_cases_0000 - lbl_cases
    extra_labels   = lbl_cases - img_cases_0000

    print(f"\n🔍  Dataset integrity check:")
    print(f"    Image cases (FLAIR) : {len(img_cases_0000)}")
    print(f"    Label cases         : {len(lbl_cases)}")

    if missing_labels:
        print(f"    ❌ Missing labels for: {missing_labels}")
    if extra_labels:
        print(f"    ⚠️  Extra labels (no matching image): {extra_labels}")

    if img_cases_0000 != lbl_cases:
        raise RuntimeError(
            f"Dataset mismatch — {len(missing_labels)} missing labels, "
            f"{len(extra_labels)} extra labels."
        )

    print("    ✅  Dataset is perfectly aligned!")
    return True