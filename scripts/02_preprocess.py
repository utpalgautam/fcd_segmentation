#!/usr/bin/env python3
"""
02_preprocess.py — Run the full preprocessing pipeline on all subjects.

Matches Joshi et al. (2025) EXACTLY:
  - FCD patients AND healthy controls both included in training
  - Single-channel FLAIR input (NO T1)
  - Top-5 axial slice selection:
      FCD patients  → slices with most ROI/lesion voxels
      Controls      → slices with highest FLAIR intensity
  - 1 mm isotropic resampling
  - Skull-strip (HD-BET or Otsu fallback)
  - Z-score normalisation within brain mask

Usage:
    python scripts/02_preprocess.py
    python scripts/02_preprocess.py --full    # override config to use all patients
    python scripts/02_preprocess.py --resume  # skip already-processed cases
"""

import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils         import load_config, setup_logger, ensure_dirs, set_nnunet_env
from src.dataset       import (
    discover_subjects, filter_labeled, check_dataset_integrity
)
from src.preprocessing import preprocess_subject

logger = setup_logger("step02")


def main():
    parser = argparse.ArgumentParser(description="Preprocess FCD MRI dataset")
    parser.add_argument("--full",   action="store_true",
                        help="Use full dataset (ignores config use_subset)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-preprocessed cases")
    args = parser.parse_args()

    cfg = load_config()
    set_nnunet_env(cfg)

    if args.full:
        cfg["use_subset"]         = False
        cfg["subset_n_patients"]  = 9999

    print("=" * 60)
    print("  Step 2: Preprocessing Pipeline")
    print("  Based on Joshi et al., Frontiers in AI, 2025")
    print("=" * 60)
    print(f"\n  Mode        : {'SUBSET (' + str(cfg['subset_n_patients']) + ' patients)' if cfg['use_subset'] else 'FULL (85 FCD + 85 Controls)'}")
    print(f"  Input       : FLAIR only (single channel — paper setting)")
    print(f"  Skull-strip : {'HD-BET' if cfg['preprocessing']['use_skull_strip'] else 'Otsu fallback'}")
    print(f"  Top-K slices: {cfg['preprocessing']['top_k_slices']} (FCD: ROI-guided; Controls: FLAIR-intensity)")
    print(f"  Target space: {cfg['preprocessing']['target_spacing']} mm\n")

    # ── Discover subjects ──────────────────────────────────────
    fcd_subjects, control_subjects = discover_subjects(
        dataset_dir=cfg["_dataset_dir"],
        use_subset=cfg["use_subset"],
        n_subset=cfg["subset_n_patients"],
    )

    # FCD: must have labels; Controls: all-zero labels (no ROI)
    valid_fcd, n_skipped = filter_labeled(fcd_subjects)
    print(f"  Labeled FCD subjects  : {len(valid_fcd)}")
    print(f"  Skipped (no label)    : {n_skipped}")
    print(f"  Healthy controls      : {len(control_subjects)} (all-zero labels)\n")

    # Paper: both FCD and controls used in training
    all_subjects = valid_fcd + control_subjects
    print(f"  Total subjects to process: {len(all_subjects)}\n")

    if not all_subjects:
        print("❌  No subjects found. Did Step 1 (download) complete?")
        sys.exit(1)

    task_dir  = cfg["_task_dir"]
    skull_dir = cfg["_skull_dir"]
    ensure_dirs(task_dir / "imagesTr", task_dir / "labelsTr", skull_dir)

    # ── Determine already-processed cases (for --resume) ──────
    already_done_cases = {}
    if args.resume:
        if (task_dir / "case_name_map.json").exists():
            with open(task_dir / "case_name_map.json") as f:
                already_done_cases = json.load(f)
        print(f"  Resume mode: {len(already_done_cases)} cases already mapped\n")

    # ── Main preprocessing loop ────────────────────────────────
    preprocessing_log = []
    case_counter      = 1
    case_name_map     = already_done_cases.copy()

    print("-" * 60)
    for entry in tqdm(all_subjects, desc="Preprocessing"):
        sid = entry["subject_id"]

        if args.resume and sid in already_done_cases:
            case_name = already_done_cases[sid]
            print(f"  [{case_counter:02d}] {sid} — SKIPPED (already done as {case_name})")
            preprocessing_log.append({
                "subject_id": sid,
                "case_id":    int(case_name.split("_")[-1]),
                "case_name":  case_name,
                "status":     "ok (resumed)",
            })
            case_counter = max(case_counter, int(case_name.split("_")[-1]) + 1)
            continue

        is_control = entry.get("label") is None
        kind = "CTRL" if is_control else "FCD "
        print(f"\n  [{case_counter:02d}/{len(all_subjects)}] {sid}  ({kind})")

        result = preprocess_subject(
            entry          = entry,
            task_dir       = task_dir,
            skull_dir      = skull_dir,
            case_id        = case_counter,
            target_spacing = tuple(cfg["preprocessing"]["target_spacing"]),
            top_k_slices   = cfg["preprocessing"]["top_k_slices"],
            use_skull_strip= cfg["preprocessing"]["use_skull_strip"],
            device         = cfg["training"]["device"],
        )

        preprocessing_log.append(result)

        if result["status"] == "ok":
            case_name_map[entry["subject_id"]] = result["case_name"]
            print(f"    ✅  Shape: {result['flair_shape']}, "
                  f"Lesion voxels: {result['lesion_voxels']}, "
                  f"Slices: {result.get('sel_idx', [])}")
        else:
            print(f"    ❌  Failed: {result.get('error', 'unknown')}")

        case_counter += 1

    # ── Save preprocessing log ─────────────────────────────────
    log_path = task_dir / "preprocessing_log.json"
    with open(log_path, "w") as f:
        json.dump(preprocessing_log, f, indent=2, default=str)
    print(f"\n  Preprocessing log → {log_path}")

    map_path = task_dir / "case_name_map.json"
    with open(map_path, "w") as f:
        json.dump(case_name_map, f, indent=2)

    # ── Integrity check ────────────────────────────────────────
    print()
    check_dataset_integrity(task_dir)

    ok_count = sum(1 for r in preprocessing_log if r["status"].startswith("ok"))
    print(f"\n  Successfully processed : {ok_count} / {len(all_subjects)}")
    print(f"  (FCD: {len(valid_fcd)}, Controls: {len(control_subjects)})")
    print("\n✅  Step 2 complete!")


if __name__ == "__main__":
    main()