#!/usr/bin/env python3
"""
07_visualize.py — Generate all figures.

Figure 1: Segmentation overlays (replicates paper Figure 4)
Figure 2: Training curves per fold (replicates paper Figure 5)

Must be run AFTER 05_inference.py (and optionally 06_evaluate.py).

Usage:
    python scripts/07_visualize.py
    python scripts/07_visualize.py --n-cases 5   # show 5 subjects instead of 3
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils     import load_config, setup_logger, set_nnunet_env
from src.visualize import plot_segmentation_overlays, plot_training_curves

logger = setup_logger("step07")


def main():
    parser = argparse.ArgumentParser(description="Generate visualisation figures")
    parser.add_argument("--n-cases", type=int, default=3,
                        help="Number of subjects to show in overlay figure (default: 3)")
    args = parser.parse_args()

    cfg = load_config()
    set_nnunet_env(cfg)

    task_dir    = cfg["_task_dir"]
    pred_dir    = cfg["_pred_dir"]
    mc_dir      = cfg["_mc_dir"]
    figures_dir = cfg["_figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Step 7: Generate Figures")
    print("=" * 60)

    # ── Load preprocessing log for case list ─────────────────
    log_path = task_dir / "preprocessing_log.json"
    if log_path.exists():
        with open(log_path) as f:
            prep_log = json.load(f)
        ok_cases = [r for r in prep_log if r.get("status", "").startswith("ok")]
    else:
        # Recover case list from imagesTr
        ok_cases = [
            {"case_name": f.name.split("_0000")[0]}
            for f in sorted((task_dir / "imagesTr").glob("*_0000.nii.gz"))
        ]

    print(f"\n  Cases available: {len(ok_cases)}")

    # ── Figure 1: Segmentation overlays ─────────────────────
    print("\n  [1/2] Generating segmentation overlays ...")
    overlay_path = plot_segmentation_overlays(
        case_list   = ok_cases,
        task_dir    = task_dir,
        pred_dir    = pred_dir,
        mc_dir      = mc_dir if mc_dir.exists() else None,
        figures_dir = figures_dir,
        n_show      = args.n_cases,
        dpi         = cfg["output"]["figure_dpi"],
    )

    # ── Figure 2: Training curves ────────────────────────────
    print("\n  [2/2] Generating training curves ...")
    curves_path = plot_training_curves(
        results_dir = cfg["_nnunet_results"],
        task_id     = cfg["nnunet"]["task_id"],
        trainer     = cfg["nnunet"]["trainer"],
        plans       = cfg["nnunet"]["plans"],
        config      = cfg["nnunet"]["config"],
        n_folds     = cfg["nnunet"]["n_folds"],
        epochs      = cfg["training"]["epochs"],
        figures_dir = figures_dir,
        dpi         = cfg["output"]["figure_dpi"],
    )

    print(f"\n  All figures saved to: {figures_dir}")
    print("\n  Files generated:")
    for f in sorted(figures_dir.glob("*.png")):
        size_kb = f.stat().st_size // 1024
        print(f"    {f.name:45s}  {size_kb:>5} KB")

    print("\n✅  Step 7 complete! Pipeline finished.\n")
    print("  Open the figures:")
    if overlay_path:
        print(f"    Overlays : {overlay_path}")
    if curves_path:
        print(f"    Curves   : {curves_path}")


if __name__ == "__main__":
    main()