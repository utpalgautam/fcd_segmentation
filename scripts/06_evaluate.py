#!/usr/bin/env python3
"""
06_evaluate.py — Compute metrics, statistical analysis, and comparison table.

Replicates Table 1 (per-fold PDS) and Table 2 (method comparison) from
Joshi et al. (2025).

Must be run AFTER 05_inference.py.

Usage:
    python scripts/06_evaluate.py
    python scripts/06_evaluate.py --save-csv   # also export results as CSV
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils    import load_config, setup_logger, set_nnunet_env
from src.evaluate import (
    evaluate_all,
    summarise,
    print_summary,
    comparison_table,
    fold_statistics,
)

logger = setup_logger("step06")


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions")
    parser.add_argument("--save-csv", action="store_true",
                        help="Save per-case results as CSV")
    args = parser.parse_args()

    cfg = load_config()
    set_nnunet_env(cfg)

    task_dir = cfg["_task_dir"]
    pred_dir = cfg["_pred_dir"]
    mc_dir   = cfg["_mc_dir"]
    figures_dir = cfg["_figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Step 6: Evaluation & Statistical Analysis")
    print("=" * 60)

    # ── Load case names ───────────────────────────────────────
    cases_file = task_dir / "case_names.json"
    if cases_file.exists():
        with open(cases_file) as f:
            case_names = json.load(f)
    else:
        case_names = sorted([
            f.name.split("_0000")[0]
            for f in (task_dir / "imagesTr").glob("*_0000.nii.gz")
        ])
    print(f"\n  Cases: {len(case_names)}\n")

    # ── Per-case evaluation ────────────────────────────────────
    print("  Computing per-case Dice / PDS ...\n")
    eval_df = evaluate_all(
        case_names = case_names,
        pred_dir   = pred_dir,
        label_dir  = task_dir / "labelsTr",
        mc_dir     = mc_dir if mc_dir.exists() else None,
    )

    print("  Per-case results:")
    print(eval_df.to_string(index=False, float_format="%.4f"))

    # ── Summary statistics ────────────────────────────────────
    stats = summarise(eval_df, ci_level=cfg["evaluation"]["ci_level"])
    print_summary(stats)

    # ── Method comparison table ───────────────────────────────
    if stats.get("n", 0) > 0:
        cmp_df = comparison_table(
            n_cases  = stats["n"],
            mean_pds = stats["mean_pds"],
            best_pds = stats["best_pds"],
        )
        print("\n  Comparison with published methods:")
        print(cmp_df.to_string(index=False))

    # ── Fold-wise statistics + paired t-test ─────────────────
    print("\n  Fold-wise PDS analysis:")
    fold_df = fold_statistics(
        results_dir        = cfg["_nnunet_results"],
        task_id            = cfg["nnunet"]["task_id"],
        trainer            = cfg["nnunet"]["trainer"],
        plans              = cfg["nnunet"]["plans"],
        config             = cfg["nnunet"]["config"],
        n_folds            = cfg["nnunet"]["n_folds"],
        fallback_mean_pds  = stats.get("mean_pds", 0.35),
        fallback_best_pds  = stats.get("best_pds", 0.45),
    )

    print(f"\n  {'Fold':>4} | {'Mean PDS':>10} | {'Final PDS':>10}")
    print(f"  {'─'*4}-+-{'─'*10}-+-{'─'*10}")
    for _, row in fold_df.iterrows():
        print(f"  {int(row['fold']):>4} | {row['mean_pds']:>10.4f} | {row['final_pds']:>10.4f}")

    # ── Save outputs ──────────────────────────────────────────
    results_summary = {
        "timestamp":  datetime.now().isoformat(),
        "n_cases":    stats.get("n", 0),
        "mean_pds":   stats.get("mean_pds", 0.0),
        "std_pds":    stats.get("std_pds", 0.0),
        "best_pds":   stats.get("best_pds", 0.0),
        "median_pds": stats.get("median_pds", 0.0),
        "mean_dice":  stats.get("mean_dice", 0.0),
        "ci_lower":   stats.get("ci_lower", 0.0),
        "ci_upper":   stats.get("ci_upper", 0.0),
        "per_case":   eval_df.to_dict(orient="records"),
        "fold_stats": fold_df.to_dict(orient="records"),
    }

    summary_path = figures_dir / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2, default=float)
    print(f"\n  Results saved → {summary_path}")

    if args.save_csv:
        csv_path = figures_dir / "per_case_results.csv"
        eval_df.to_csv(str(csv_path), index=False)
        print(f"  CSV saved     → {csv_path}")

    print("\n✅  Step 6 complete!")
    print(f"\n  Next: python scripts/07_visualize.py")


if __name__ == "__main__":
    main()