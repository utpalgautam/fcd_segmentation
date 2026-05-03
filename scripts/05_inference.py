#!/usr/bin/env python3
"""
05_inference.py — Run nnU-Net ensemble inference + MC-Dropout uncertainty.

Two phases:
  A. Standard ensemble inference via nnUNetv2_predict (all 5 folds)
  B. MC-Dropout inference: T=20 stochastic forward passes per case

Paper (Joshi et al., 2025):
  - Input: single-channel FLAIR (top-5 axial slices)
  - Ensemble: all 5 fold checkpoints (checkpoint_best.pth)
  - MC-Dropout: T=20 samples, p=0.2

Must be run AFTER 04_train.py.

Usage:
    python scripts/05_inference.py
    python scripts/05_inference.py --no-mc     # skip MC-Dropout
    python scripts/05_inference.py --mc-only   # skip standard inference
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils     import load_config, setup_logger, ensure_dirs, set_nnunet_env
from src.inference import (
    run_nnunet_inference,
    load_nnunet_predictor,
    run_mc_inference_for_case,
)
from src.train import get_checkpoint_name

logger = setup_logger("step05")


def main():
    parser = argparse.ArgumentParser(description="Run inference (Joshi et al., 2025)")
    parser.add_argument("--no-mc",   action="store_true",
                        help="Skip MC-Dropout inference")
    parser.add_argument("--mc-only", action="store_true",
                        help="Skip standard nnU-Net inference")
    args = parser.parse_args()

    cfg = load_config()
    set_nnunet_env(cfg)

    task_id  = cfg["nnunet"]["task_id"]
    config   = cfg["nnunet"]["config"]
    trainer  = cfg["nnunet"]["trainer"]
    plans    = cfg["nnunet"]["plans"]
    device   = cfg["training"]["device"]
    mc_cfg   = cfg["mc_dropout"]
    task_dir = cfg["_task_dir"]
    pred_dir = cfg["_pred_dir"]
    mc_dir   = cfg["_mc_dir"]

    print("=" * 60)
    print("  Step 5: Inference + MC-Dropout Uncertainty")
    print("  Paper: Joshi et al., Frontiers in AI, 2025")
    print("=" * 60)
    print(f"\n  Task ID    : {task_id}")
    print(f"  Config     : {config}")
    print(f"  Trainer    : {trainer}")
    print(f"  Input      : Single-channel FLAIR (top-5 slices)")
    print(f"  MC samples : {mc_cfg['n_samples']} (paper: 20)")
    print(f"  Dropout p  : {mc_cfg['dropout_p']} (paper: 0.2)")
    print(f"  Conf thresh: {mc_cfg['confidence_thresh']}")
    print(f"  Device     : {device}\n")

    # Load case names
    cases_file = task_dir / "case_names.json"
    if not cases_file.exists():
        case_names = sorted([
            f.name.split("_0000")[0]
            for f in (task_dir / "imagesTr").glob("*_0000.nii.gz")
        ])
    else:
        with open(cases_file) as f:
            case_names = json.load(f)

    print(f"  Cases to process: {len(case_names)}\n")

    # Discover trained folds
    model_base_dir = (
        cfg["_nnunet_results"]
        / f"Dataset{task_id:03d}_FCD"
        / f"{trainer}__{plans}__{config}"
    )
    available_folds = []
    if model_base_dir.exists():
        for fold_dir in model_base_dir.glob("fold_*"):
            if (fold_dir / "checkpoint_best.pth").exists():
                available_folds.append(fold_dir.name.split("_")[1])

    folds_str = ",".join(sorted(available_folds)) if available_folds else "0"
    print(f"  Trained folds detected: {folds_str}")

    # ── Phase A: Standard nnU-Net ensemble inference ──────────
    if not args.mc_only:
        print("─" * 60)
        print("  Phase A: Standard nnU-Net Ensemble Inference (all folds)")
        print("─" * 60)
        ensure_dirs(pred_dir)

        if not available_folds:
            print("  ⚠️  No trained folds found. Skipping Phase A.")
            success = False
        else:
            success = run_nnunet_inference(
                input_dir          = task_dir / "imagesTr",
                output_dir         = pred_dir,
                task_id            = task_id,
                config             = config,
                folds              = folds_str,
                save_probabilities = cfg["output"]["save_probabilities"],
                checkpoint_name    = "checkpoint_best.pth",
                trainer            = trainer,
                plans              = plans,
            )

        if success:
            n_preds = len(list(pred_dir.glob("*.nii.gz")))
            print(f"  ✅  {n_preds} prediction files in {pred_dir}")
        else:
            print("  ⚠️  Standard inference had errors; check output above.")

    # ── Phase B: MC-Dropout inference ─────────────────────────
    if not args.no_mc:
        print("\n" + "─" * 60)
        print("  Phase B: MC-Dropout Uncertainty Inference")
        print(f"    T={mc_cfg['n_samples']} forward passes, p={mc_cfg['dropout_p']} dropout")
        print("─" * 60)
        ensure_dirs(mc_dir)

        try:
            ckpt = get_checkpoint_name(
                cfg["_nnunet_results"], task_id, trainer, plans, config, fold=0
            )
            print(f"  Loading checkpoint: {ckpt} (fold 0)")

            predictor = load_nnunet_predictor(
                results_dir    = cfg["_nnunet_results"],
                task_id        = task_id,
                trainer        = trainer,
                plans          = plans,
                config         = config,
                fold           = 0,
                checkpoint_name= ckpt,
                device         = device,
            )

            mc_results = []
            for case_name in tqdm(case_names, desc="MC-Dropout"):
                result = run_mc_inference_for_case(
                    case_name = case_name,
                    task_dir  = task_dir,
                    mc_out_dir= mc_dir,
                    predictor = predictor,
                    mc_cfg    = mc_cfg,
                    device    = device,
                )
                mc_results.append(result)

                if result.get("pds") is not None:
                    print(f"    {case_name}: PDS={result['pds']:.4f}  "
                          f"Dice={result['dice']:.4f}  "
                          f"Entropy={result['mean_entropy']:.4f}")

            # Save MC results
            mc_results_path = mc_dir / "mc_results.json"
            with open(mc_results_path, "w") as f:
                json.dump(mc_results, f, indent=2, default=float)
            print(f"\n  MC results saved → {mc_results_path}")

        except FileNotFoundError as exc:
            print(f"\n  ⚠️  Could not load model: {exc}")
            print("  Skipping MC-Dropout. Standard predictions will be used for evaluation.")
        except Exception as exc:
            print(f"\n  ⚠️  MC-Dropout failed: {exc}")
            logger.exception("MC-Dropout error")

    print("\n✅  Step 5 complete!")
    print(f"\n  Next: python scripts/06_evaluate.py")


if __name__ == "__main__":
    main()