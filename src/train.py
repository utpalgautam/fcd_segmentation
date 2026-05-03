"""
train.py — nnU-Net training wrapper.

Wraps nnUNetv2_plan_and_preprocess and nnUNetv2_train CLI commands.

Paper (Joshi et al., 2025) settings:
  - Trainer: nnUNetTrainer_100epochs (custom 100-epoch trainer)
  - Config:  3d_fullres
  - Folds:   5-fold cross-validation
  - Optimizer: SGD, momentum=0.99, Nesterov, LR=0.01
  - Loss: Dice + CE (nnU-Net default, combined loss)
"""

import os
import time
import subprocess
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("fcd.train")


# ─────────────────────────────────────────────────────────────
# 1.  Fingerprinting + planning
# ─────────────────────────────────────────────────────────────

def plan_and_preprocess(
    task_id: int,
    config: str = "3d_fullres",
    verify_integrity: bool = True,
) -> bool:
    """
    Run nnUNetv2_plan_and_preprocess for the given task.
    Computes the dataset fingerprint and generates the training plan.
    """
    import shutil
    import sys
    python_dir = os.path.dirname(sys.executable)
    cmd_bin    = os.path.join(python_dir, "nnUNetv2_plan_and_preprocess")

    if not os.path.exists(cmd_bin):
        cmd_bin = shutil.which("nnUNetv2_plan_and_preprocess") or "nnUNetv2_plan_and_preprocess"

    cmd = [
        cmd_bin,
        "-d", str(task_id),
        "-c", config,
    ]
    if verify_integrity:
        cmd.append("--verify_dataset_integrity")

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd)

    if result.returncode == 0:
        logger.info("✅  Fingerprinting and planning complete")
        return True
    else:
        logger.warning("⚠️  Fingerprinting returned code %d", result.returncode)
        return False


# ─────────────────────────────────────────────────────────────
# 2.  Single-fold training
# ─────────────────────────────────────────────────────────────

def train_fold(
    task_id: int,
    config: str,
    fold: int,
    trainer: str = "nnUNetTrainer_100epochs",
    plans: str = "nnUNetPlans",
    device: str = "cuda",
    extra_args: Optional[list] = None,
) -> dict:
    """
    Train nnU-Net for a single fold.

    Paper: 100 epochs, 3d_fullres, 5-fold CV, SGD momentum=0.99 LR=0.01.
    The custom trainer (nnUNetTrainer_100epochs) encodes these settings.

    Returns a dict with keys: fold, status, elapsed_min.
    """
    t_start = time.time()
    status  = "failed"

    import shutil
    import sys
    python_dir = os.path.dirname(sys.executable)
    cmd_bin    = os.path.join(python_dir, "nnUNetv2_train")

    if not os.path.exists(cmd_bin):
        cmd_bin = shutil.which("nnUNetv2_train") or "nnUNetv2_train"

    cmd = [
        cmd_bin,
        str(task_id),
        config,
        str(fold),
        "-tr", trainer,
        "-p", plans,
        "--npz",           # save softmax predictions (needed for ensembling)
    ]

    # Device flag
    if device == "cpu":
        cmd += ["--device", "cpu"]

    if extra_args:
        cmd += extra_args

    logger.info("Fold %d | Running: %s", fold, " ".join(cmd))

    try:
        result = subprocess.run(cmd)
        status = "ok" if result.returncode == 0 else "error"
    except KeyboardInterrupt:
        logger.warning("Fold %d interrupted by user", fold)
        status = "interrupted"
    except Exception as exc:
        logger.error("Fold %d crashed: %s", fold, exc)
        status = "crashed"

    elapsed = (time.time() - t_start) / 60.0
    logger.info("Fold %d — %s (%.1f min)", fold, status.upper(), elapsed)

    return {"fold": fold, "status": status, "elapsed_min": elapsed}


# ─────────────────────────────────────────────────────────────
# 3.  Full 5-fold cross-validation
# ─────────────────────────────────────────────────────────────

def train_all_folds(
    task_id: int,
    config: str = "3d_fullres",
    n_folds: int = 5,
    trainer: str = "nnUNetTrainer_100epochs",
    plans: str = "nnUNetPlans",
    device: str = "cuda",
) -> list:
    """
    Sequentially train all N folds.
    Paper: 5-fold cross-validation with 100 epochs each.
    Returns a list of per-fold result dicts.
    """
    results = []
    for fold in range(n_folds):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold + 1} / {n_folds}")
        print(f"{'='*60}")

        fold_result = train_fold(
            task_id  = task_id,
            config   = config,
            fold     = fold,
            trainer  = trainer,
            plans    = plans,
            device   = device,
        )
        results.append(fold_result)

    # Print summary
    print("\n\nTraining Summary:")
    print(f"  {'Fold':>6} | {'Status':>12} | {'Time (min)':>10}")
    print(f"  {'─'*6}-+-{'─'*12}-+-{'─'*10}")
    for r in results:
        print(f"  {r['fold'] + 1:>6} | {r['status']:>12} | {r['elapsed_min']:>10.1f}")

    return results


# ─────────────────────────────────────────────────────────────
# 4.  Determine best checkpoint name
# ─────────────────────────────────────────────────────────────

def get_checkpoint_name(
    results_dir: Path,
    task_id: int,
    trainer: str,
    plans: str,
    config: str,
    fold: int = 0,
) -> str:
    """
    Return 'checkpoint_best.pth' if it exists in the fold directory,
    else 'checkpoint_final.pth'.
    """
    fold_dir = (
        results_dir
        / f"Dataset{task_id:03d}_FCD"
        / f"{trainer}__{plans}__{config}"
        / f"fold_{fold}"
    )
    if (fold_dir / "checkpoint_best.pth").exists():
        return "checkpoint_best.pth"
    return "checkpoint_final.pth"