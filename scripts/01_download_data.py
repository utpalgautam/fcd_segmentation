#!/usr/bin/env python3
"""
01_download_data.py — Download the OpenNeuro ds004199 dataset.

Dataset: "An open presurgery MRI dataset of people with epilepsy and FCD type II"
Schuch et al., Scientific Data, 2023
OpenNeuro: https://openneuro.org/datasets/ds004199

Usage:
    python scripts/01_download_data.py
    python scripts/01_download_data.py --check-only   # verify without downloading
"""

import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_config, setup_logger

logger = setup_logger("step01")


def main():
    parser = argparse.ArgumentParser(description="Download OpenNeuro ds004199")
    parser.add_argument("--check-only", action="store_true",
                        help="Only check if data exists; do not download")
    args = parser.parse_args()

    cfg         = load_config()
    dataset_dir = cfg["_dataset_dir"]
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Step 1: Dataset Download")
    print("  OpenNeuro ds004199 — FCD Type II MRI")
    print("=" * 60)
    print(f"\n  Target directory : {dataset_dir}")
    print(f"  Estimated size   : ~20 GB")
    print(f"  Estimated time   : 30–90 min (depends on connection)\n")

    # ── Check if already downloaded ───────────────────────────
    subjects = sorted([d for d in dataset_dir.iterdir()
                       if d.is_dir() and d.name.startswith("sub-")])

    if subjects:
        print(f"  Found {len(subjects)} subject directories.")
        flair_count = sum(1 for s in subjects
                          for _ in s.rglob("*FLAIR*.nii*"))
        print(f"  Found {flair_count} FLAIR files.")

        if args.check_only:
            print("\n  ✅  Dataset appears to be present.")
            return

        if len(subjects) >= 80:
            print("\n  Dataset appears complete. Skipping download.")
            print("  (Delete outputs/dataset/ and rerun to force re-download)")
            return

    if args.check_only:
        print("\n  ❌  Dataset not found.")
        print(f"      Expected at: {dataset_dir}")
        return

    # ── Run download ───────────────────────────────────────────
    print("  Starting download via openneuro-py ...\n")
    print("  ⚠️  This will take 30–90 min and requires ~20 GB disk space.")
    print("  ⚠️  If interrupted, re-run this script — it resumes automatically.\n")

    # Try to find openneuro-py in the same folder as the current python
    import shutil
    import os
    python_dir    = os.path.dirname(sys.executable)
    openneuro_bin = os.path.join(python_dir, "openneuro-py")
    
    if not os.path.exists(openneuro_bin):
        openneuro_bin = shutil.which("openneuro-py") or "openneuro-py"

    result = subprocess.run([
        openneuro_bin, "download",
        "--dataset", cfg["dataset"]["openneuro_id"],
        "--target-dir", str(dataset_dir),
    ])

    if result.returncode != 0:
        print(f"\n  ⚠️  openneuro-py exited with code {result.returncode}")
        print("  This can happen after partial downloads — check the data directory.")

    # ── Post-download verification ─────────────────────────────
    subjects = sorted([d for d in dataset_dir.iterdir()
                       if d.is_dir() and d.name.startswith("sub-")])

    print(f"\n  Download summary:")
    print(f"    Subjects found : {len(subjects)}")

    if subjects:
        example = subjects[0]
        flair = list(example.rglob("*FLAIR*.nii*"))
        t1    = list(example.rglob("*T1w*.nii*"))
        roi   = list(example.rglob("*roi*.nii*"))
        print(f"    Example subject: {example.name}")
        print(f"      FLAIR : {flair[0].name if flair else '❌ NOT FOUND'}")
        print(f"      T1    : {t1[0].name    if t1    else '❌ NOT FOUND'}")
        print(f"      ROI   : {roi[0].name   if roi   else 'NOT FOUND (check derivatives/)'}")

    ptsv = dataset_dir / "participants.tsv"
    print(f"\n    participants.tsv : {'✅ found' if ptsv.exists() else '❌ NOT FOUND'}")

    print("\n✅  Step 1 complete!")


if __name__ == "__main__":
    main()