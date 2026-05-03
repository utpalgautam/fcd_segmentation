#!/usr/bin/env python3
"""
03_prepare_nnunet.py — Build dataset.json and run nnU-Net fingerprinting.

Must be run AFTER 02_preprocess.py.

Paper (Joshi et al., 2025):
  - 85 FCD patients + 85 healthy controls = 170 total cases
  - Single-channel FLAIR input
  - 5-fold cross-validation
  - nnUNetTrainer_100epochs (custom trainer, 100 epochs)

Usage:
    python scripts/03_prepare_nnunet.py
"""

import sys
import json
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils   import load_config, setup_logger, set_nnunet_env
from src.dataset import write_dataset_json, check_dataset_integrity
from src.train   import plan_and_preprocess

logger = setup_logger("step03")


def main():
    cfg = load_config()
    set_nnunet_env(cfg)

    task_dir  = cfg["_task_dir"]
    task_id   = cfg["nnunet"]["task_id"]
    task_name = cfg["_task_name"]
    trainer   = cfg["nnunet"]["trainer"]

    print("=" * 60)
    print("  Step 3: Prepare nnU-Net Dataset + Fingerprinting")
    print("  Based on Joshi et al., Frontiers in AI, 2025")
    print("=" * 60)
    print(f"\n  Task ID    : {task_id}")
    print(f"  Task Name  : {task_name}")
    print(f"  Config     : {cfg['nnunet']['config']}")
    print(f"  Trainer    : {trainer}")
    print(f"  Input      : Single-channel FLAIR (paper setting)")
    print(f"  N folds    : {cfg['nnunet']['n_folds']}\n")

    # ── Verify preprocessing was done ────────────────────────
    if not (task_dir / "imagesTr").exists():
        print("❌  imagesTr/ not found. Run 02_preprocess.py first.")
        sys.exit(1)

    check_dataset_integrity(task_dir)

    # ── Collect case names ────────────────────────────────────
    case_names = sorted([
        f.name.split("_0000")[0]
        for f in (task_dir / "imagesTr").glob("*_0000.nii.gz")
    ])
    print(f"\n  Cases found: {len(case_names)}")
    if not case_names:
        print("  ❌  No cases found in imagesTr/. Check preprocessing step.")
        sys.exit(1)

    print("  First few cases:", case_names[:3])

    # ── Write dataset.json (single-channel FLAIR) ─────────────
    print("\n  Writing dataset.json (single-channel FLAIR) ...")
    write_dataset_json(
        task_dir=task_dir,
        case_names=case_names,
        task_name=task_name,
        seed=cfg["evaluation"]["random_seed"],
    )

    # Save case list for downstream scripts
    cases_file = task_dir / "case_names.json"
    with open(cases_file, "w") as f:
        json.dump(case_names, f, indent=2)
    print(f"  Case list → {cases_file}")

    # ── Register custom trainer with nnU-Net ──────────────────
    # The custom trainer class (nnUNetTrainer_100epochs) must be
    # importable by nnU-Net. We ensure the src/ directory is on PYTHONPATH.
    _install_trainer_if_needed(trainer)

    # ── Run nnU-Net fingerprinting ────────────────────────────
    print("\n  Running nnU-Net fingerprinting and planning ...")
    print("  (Analyses dataset to auto-configure architecture)\n")

    success = plan_and_preprocess(
        task_id          = task_id,
        config           = cfg["nnunet"]["config"],
        verify_integrity = True,
    )

    if not success:
        print("\n  ⚠️  Fingerprinting may have had warnings.")
        plan_dir = cfg["_nnunet_prep"] / task_name
        if plan_dir.exists():
            plan_files = list(plan_dir.glob("*.json"))
            if plan_files:
                print(f"  Plans found: {[f.name for f in plan_files]}")
                print("  Continuing ...")
            else:
                print("  ❌  No plan files generated. Check nnU-Net output above.")
                sys.exit(1)

    print("\n✅  Step 3 complete!")
    print(f"\n  Next: python scripts/04_train.py")


def _install_trainer_if_needed(trainer_name: str) -> None:
    """
    Ensure the custom trainer class is discoverable by nnU-Net.
    Copies the trainer file to nnU-Net's trainer directory.
    """
    import shutil
    import importlib
    import os

    # Check if already importable
    try:
        importlib.import_module(f"src.nnunet_trainer_100ep")
        print(f"  ✅  Custom trainer '{trainer_name}' found in src/")
    except ImportError:
        pass

    # Try to find nnU-Net trainer directory and copy our custom trainer
    try:
        import nnunetv2
        nnunet_trainer_dir = Path(nnunetv2.__file__).parent / "training" / "nnUNetTrainer"
        custom_trainer_src = Path(__file__).parent.parent / "src" / "nnunet_trainer_100ep.py"

        if custom_trainer_src.exists() and nnunet_trainer_dir.exists():
            dst = nnunet_trainer_dir / "nnUNetTrainer_100epochs.py"
            shutil.copy(custom_trainer_src, dst)
            print(f"  ✅  Custom trainer copied → {dst}")
        else:
            print(f"  ⚠️  Could not install custom trainer to nnU-Net directory.")
            print(f"      Training will use trainer: {trainer_name}")
            print(f"      If this fails, set trainer: nnUNetTrainer in configs/config.yaml")
    except Exception as exc:
        print(f"  ⚠️  Trainer installation warning: {exc}")


if __name__ == "__main__":
    main()