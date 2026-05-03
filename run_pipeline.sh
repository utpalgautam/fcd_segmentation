#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — Master script: runs the entire FCD segmentation pipeline
# Usage: ./run_pipeline.sh [--skip-download] [--skip-preprocess] [--full]
# =============================================================================

set -e

export PYTHONUNBUFFERED=1

SKIP_DOWNLOAD=false
SKIP_PREPROCESS=false
FULL_DATASET=false
START_FROM=1

# Parse arguments
for arg in "$@"; do
    case $arg in
        --skip-download)   SKIP_DOWNLOAD=true ;;
        --skip-preprocess) SKIP_PREPROCESS=true ;;
        --full)            FULL_DATASET=true ;;
        --from=*)          START_FROM="${arg#*=}" ;;
        --help)
            echo "Usage: ./run_pipeline.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-download    Skip OpenNeuro download (data already exists)"
            echo "  --skip-preprocess  Skip preprocessing (already done)"
            echo "  --full             Use full dataset (85 patients, overrides config)"
            echo "  --from=N           Start from step N (1–7)"
            echo "  --help             Show this help"
            exit 0
            ;;
    esac
done

# ── Activate Python environment ───────────────────────────────
source venv_download/bin/activate 2>/dev/null || echo "⚠️  Could not activate venv_download"

echo "============================================================"
echo "  FCD Type II Segmentation Pipeline"
echo "  Based on Joshi et al., Frontiers in AI, 2025"
echo "============================================================"
echo ""
echo "  Mode         : $([ '$FULL_DATASET' = true ] && echo 'FULL (85 patients)' || echo 'SUBSET (10 patients)')"
echo "  Starting from: Step $START_FROM"
echo ""

PYTHON="python"
LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

run_step() {
    STEP_NUM=$1
    STEP_NAME=$2
    SCRIPT=$3
    EXTRA_ARGS="${4:-}"

    if [ "$STEP_NUM" -lt "$START_FROM" ]; then
        echo "⏭️  Step $STEP_NUM: $STEP_NAME — SKIPPED (--from=$START_FROM)"
        return
    fi

    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo "  Step $STEP_NUM / 7 : $STEP_NAME"
    echo "══════════════════════════════════════════════════════════"

    LOG_FILE="$LOG_DIR/step${STEP_NUM}_${TIMESTAMP}.log"
    START_TIME=$(date +%s)

    if $PYTHON "$SCRIPT" $EXTRA_ARGS 2>&1 | tee "$LOG_FILE"; then
        END_TIME=$(date +%s)
        ELAPSED=$(( (END_TIME - START_TIME) / 60 ))
        echo ""
        echo "  ✅ Step $STEP_NUM complete — ${ELAPSED} min"
        echo "  Log: $LOG_FILE"
    else
        echo ""
        echo "  ❌ Step $STEP_NUM FAILED. Check: $LOG_FILE"
        echo "  To resume from this step: ./run_pipeline.sh --from=$STEP_NUM"
        exit 1
    fi
}

# Optional flags for scripts
FULL_FLAG=""
if [ "$FULL_DATASET" = true ]; then FULL_FLAG="--full"; fi

# ── Run all steps ─────────────────────────────────────────────

if [ "$SKIP_DOWNLOAD" = false ]; then
    run_step 1 "Download Dataset (OpenNeuro ds004199)" \
        "scripts/01_download_data.py"
else
    echo "⏭️  Step 1: Download — SKIPPED (--skip-download)"
fi

if [ "$SKIP_PREPROCESS" = false ]; then
    run_step 2 "Preprocessing (Registration + Skull-strip + Normalize)" \
        "scripts/02_preprocess.py" "$FULL_FLAG"
else
    echo "⏭️  Step 2: Preprocessing — SKIPPED (--skip-preprocess)"
fi

run_step 3 "Prepare nnU-Net Dataset + Fingerprinting" \
    "scripts/03_prepare_nnunet.py"

run_step 4 "Train nnU-Net (5-Fold Cross-Validation)" \
    "scripts/04_train.py"

run_step 5 "Inference + MC-Dropout Uncertainty Maps" \
    "scripts/05_inference.py"

run_step 6 "Evaluation + Statistical Analysis" \
    "scripts/06_evaluate.py"

run_step 7 "Generate Figures" \
    "scripts/07_visualize.py"

# ── Final summary ─────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  ✅ PIPELINE COMPLETE!"
echo "============================================================"
echo ""
echo "  Output files:"
echo "    Predictions   : outputs/work/predictions/"
echo "    MC-Dropout    : outputs/work/mc_dropout_predictions/"
echo "    Figures       : outputs/figures/"
echo "    Logs          : outputs/logs/"
echo ""
echo "  To view results: open outputs/figures/ in your file browser"
echo ""