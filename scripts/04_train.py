#!/usr/bin/env python3
"""
04_train.py — Train nnU-Net with 5-fold cross-validation.

Paper (Joshi et al., 2025) settings:
  - Trainer: nnUNetTrainer_100epochs (100 epochs, SGD LR=0.01, momentum=0.99)
  - Config:  3d_fullres
  - Folds:   5-fold cross-validation
  - Loss:    Dice + Cross-Entropy (nnU-Net default combined loss)
  - Input:   Single-channel FLAIR (top-5 axial slices per subject)

Expected time:
    100 epochs × 5 folds : ~12–48 hrs (GPU, depends on VRAM)

Usage:
    python scripts/04_train.py
    python scripts/04_train.py --fold 0        # train only fold 0
    python scripts/04_train.py --folds 0 1 2   # train specific folds
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_config, setup_logger, set_nnunet_env
from src.train import train_all_folds, train_fold

logger = setup_logger("step04")


def main():
    parser = argparse.ArgumentParser(description="Train nnU-Net (Joshi et al., 2025)")
    parser.add_argument("--fold",  type=int, default=None,
                        help="Train a single fold (0-indexed)")
    parser.add_argument("--folds", type=int, nargs="+", default=None,
                        help="Train specific folds (e.g. --folds 0 1 2)")
    args = parser.parse_args()

    cfg = load_config()
    set_nnunet_env(cfg)

    task_id  = cfg["nnunet"]["task_id"]
    config   = cfg["nnunet"]["config"]
    trainer  = cfg["nnunet"]["trainer"]   # nnUNetTrainer_100epochs
    plans    = cfg["nnunet"]["plans"]
    n_folds  = cfg["nnunet"]["n_folds"]
    epochs   = cfg["training"]["epochs"]
    device   = cfg["training"]["device"]

    print("=" * 60)
    print("  Step 4: nnU-Net Training")
    print("  Paper: Joshi et al., Frontiers in AI, 2025")
    print("=" * 60)
    print(f"\n  Task ID   : {task_id}")
    print(f"  Config    : {config}  (3D full-resolution)")
    print(f"  Trainer   : {trainer}")
    print(f"  Epochs    : {epochs}  (paper: 100)")
    print(f"  Folds     : {n_folds}-fold cross-validation")
    print(f"  Optimizer : SGD, LR={cfg['training']['learning_rate']}, momentum={cfg['training']['momentum']}")
    print(f"  Loss      : Dice + Cross-Entropy (nnU-Net default)")
    print(f"  Input     : Single-channel FLAIR (top-5 slices)")
    print(f"  Device    : {device}")

    if epochs < 100:
        print(f"\n  ⚠️  NOTICE: epochs={epochs} is set for quick testing.")
        print(f"             Set epochs: 100 in configs/config.yaml for paper results.")

    print(f"\n  ─── Estimated time ───────────────────────────────")
    if device == "cpu":
        print(f"  Very slow on CPU — consider GPU or reduce epochs.")
    else:
        print(f"  ~{epochs * n_folds * 2 // 60 + 1}–{epochs * n_folds * 5 // 60 + 1} hrs (GPU)")
    print()

    # ── Ensure custom trainer is installed ──────────────────
    _ensure_custom_trainer(trainer)

    # ── Determine which folds to train ───────────────────────
    if args.fold is not None:
        folds_to_train = [args.fold]
    elif args.folds is not None:
        folds_to_train = args.folds
    else:
        folds_to_train = list(range(n_folds))

    print(f"  Training folds: {folds_to_train}\n")

    # ── nnU-Net environment tweaks ───────────────────────────
    import os
    os.environ["nnUNet_n_proc_DA"] = str(min(4, os.cpu_count() or 2))

    results = []
    for fold in folds_to_train:
        result = train_fold(
            task_id  = task_id,
            config   = config,
            fold     = fold,
            trainer  = trainer,
            plans    = plans,
            device   = device,
        )
        results.append(result)

        if result["status"] not in ("ok", "interrupted"):
            print(f"\n  ❌  Fold {fold} failed. Check nnU-Net output above.")
            print(f"  To retry this fold: python scripts/04_train.py --fold {fold}")
            # Try with fallback trainer on first failure
            if trainer == "nnUNetTrainer_100epochs":
                print(f"\n  ℹ️  If trainer not found, try setting trainer: nnUNetTrainer")
                print(f"     in configs/config.yaml (will use 1000 epochs instead of 100)")

    # ── Summary ───────────────────────────────────────────────
    ok_folds     = [r for r in results if r["status"] in ("ok", "interrupted")]
    failed_folds = [r for r in results if r["status"] not in ("ok", "interrupted")]

    print("\n" + "=" * 60)
    print(f"  Training complete: {len(ok_folds)}/{len(results)} folds OK")
    if failed_folds:
        print(f"  ❌  Failed folds: {[r['fold'] for r in failed_folds]}")
    total_min = sum(r["elapsed_min"] for r in results)
    print(f"  Total training time: {total_min:.1f} min")
    print("=" * 60)

    print("\n✅  Step 4 complete!")
    print(f"\n  Next: python scripts/05_inference.py")


def _ensure_custom_trainer(trainer_name: str) -> None:
    """Copy custom trainer to nnU-Net directory if not already there."""
    if trainer_name == "nnUNetTrainer":
        return  # Using default trainer, nothing to do

    import shutil
    try:
        import nnunetv2
        nnunet_trainer_dir = Path(nnunetv2.__file__).parent / "training" / "nnUNetTrainer"
        custom_trainer_src = Path(__file__).parent.parent / "src" / "nnunet_trainer_100ep.py"
        dst = nnunet_trainer_dir / "nnUNetTrainer_100epochs.py"

        if not dst.exists() and custom_trainer_src.exists():
            shutil.copy(custom_trainer_src, dst)
            print(f"  ✅  Custom trainer installed → {dst}")
        elif dst.exists():
            print(f"  ✅  Custom trainer already installed")
        else:
            print(f"  ⚠️  Custom trainer src not found at {custom_trainer_src}")
    except Exception as exc:
        print(f"  ⚠️  Could not install custom trainer: {exc}")
        print(f"      Training may fail if '{trainer_name}' is not found.")
        print(f"      Fallback: set trainer: nnUNetTrainer in config.yaml")


if __name__ == "__main__":
    main()