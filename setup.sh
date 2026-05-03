#!/usr/bin/env bash
# =============================================================================
# setup.sh — One-time environment setup for FCD Segmentation Pipeline
# Run: chmod +x setup.sh && ./setup.sh
# =============================================================================

set -e  # Exit on any error

ENV_NAME="fcd_seg"
PYTHON_VERSION="3.10"

echo "============================================================"
echo "  FCD Segmentation Pipeline — Environment Setup"
echo "============================================================"
echo ""

# ── Check for conda ────────────────────────────────────────────
if ! command -v conda &> /dev/null; then
    echo "❌ conda not found. Please install Miniconda first:"
    echo "   https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo "✅ conda found: $(conda --version)"

# ── Create conda environment ──────────────────────────────────
echo ""
echo "[1/7] Creating conda environment '${ENV_NAME}' (Python ${PYTHON_VERSION})..."
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y 2>/dev/null || \
    echo "  (Environment already exists — skipping creation)"

# ── Activate environment ──────────────────────────────────────
echo ""
echo "[2/7] Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
echo "  Active env: $CONDA_DEFAULT_ENV"

# ── Install PyTorch with CUDA ─────────────────────────────────
echo ""
echo "[3/7] Installing PyTorch (CUDA 12.1)..."
echo "  Checking CUDA availability..."

if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1)
    echo "  GPU detected. CUDA version: $CUDA_VERSION"

    if [ "$CUDA_VERSION" -ge "12" ] 2>/dev/null; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
        echo "  ✅ PyTorch with CUDA 12.1 installed"
    elif [ "$CUDA_VERSION" -ge "11" ] 2>/dev/null; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
        echo "  ✅ PyTorch with CUDA 11.8 installed"
    else
        echo "  ⚠️ CUDA version too old. Installing CPU-only PyTorch."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
    fi
else
    echo "  No GPU detected. Installing CPU-only PyTorch."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
    echo "  ✅ PyTorch (CPU) installed"
fi

# ── Install nnU-Net ───────────────────────────────────────────
echo ""
echo "[4/7] Installing nnU-Net v2..."
pip install nnunetv2 -q
echo "  ✅ nnU-Net v2 installed"

# ── Install custom 100-epoch trainer ─────────────────────────
echo ""
echo "  Installing custom nnUNetTrainer_100epochs (paper: 100 epochs)..."
NNUNET_TRAINER_DIR=$(python -c "import nnunetv2; from pathlib import Path; print(Path(nnunetv2.__file__).parent / 'training' / 'nnUNetTrainer')" 2>/dev/null)
if [ -n "$NNUNET_TRAINER_DIR" ] && [ -d "$NNUNET_TRAINER_DIR" ]; then
    cp src/nnunet_trainer_100ep.py "$NNUNET_TRAINER_DIR/nnUNetTrainer_100epochs.py"
    echo "  ✅ Custom trainer installed → $NNUNET_TRAINER_DIR/nnUNetTrainer_100epochs.py"
else
    echo "  ⚠️  Could not find nnU-Net trainer directory — will install during step 3"
fi

# ── Install medical imaging libraries ────────────────────────
echo ""
echo "[5/7] Installing medical imaging libraries..."
pip install nibabel SimpleITK nilearn -q
echo "  ✅ nibabel, SimpleITK, nilearn installed"

# ANTsPy (large package, may take a while)
echo "  Installing antspyx (this may take 5–10 minutes)..."
pip install antspyx -q || echo "  ⚠️ antspyx failed — will use SimpleITK for registration"
echo "  ✅ antspyx installed (or skipped)"

# ── Install HD-BET for skull stripping ───────────────────────
echo ""
echo "[6/7] Installing HD-BET (skull stripping)..."
pip install git+https://github.com/MIC-DKFZ/HD-BET.git -q || \
    echo "  ⚠️ HD-BET install failed — Otsu fallback will be used"
echo "  ✅ HD-BET installed (or skipped)"

# ── Install remaining dependencies ───────────────────────────
echo ""
echo "[7/7] Installing remaining dependencies..."
pip install \
    openneuro-py \
    scipy scikit-learn pandas numpy \
    matplotlib seaborn tqdm \
    PyYAML \
    -q
echo "  ✅ All dependencies installed"

# ── Verify installation ───────────────────────────────────────
echo ""
echo "============================================================"
echo "  Verifying Installation"
echo "============================================================"

python -c "
import torch
print(f'  PyTorch     : {torch.__version__}')
print(f'  CUDA avail  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU         : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

import nnunetv2
print(f'  nnU-Net v2  : OK')

import nibabel, SimpleITK
print(f'  nibabel     : {nibabel.__version__}')
print(f'  SimpleITK   : {SimpleITK.Version()}')

try:
    import HD_BET
    print(f'  HD-BET      : OK')
except:
    print(f'  HD-BET      : Not available (Otsu fallback active)')
"

# ── Create output directories ────────────────────────────────
echo ""
echo "Creating output directories..."
mkdir -p outputs/{dataset,figures}
mkdir -p outputs/work/{nnUNet_raw,nnUNet_preprocessed,nnUNet_results,skull_stripped,predictions,mc_dropout_predictions}
echo "  ✅ Output directories created"

echo ""
echo "============================================================"
echo "  ✅ Setup Complete!"
echo "============================================================"
echo ""
echo "  NEXT STEPS:"
echo "  1. Activate env   : conda activate ${ENV_NAME}"
echo "  2. Run pipeline   : ./run_pipeline.sh"
echo "  3. Or step-by-step: python scripts/01_download_data.py"
echo ""