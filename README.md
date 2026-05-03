# FCD Type II Lesion Segmentation — Local Pipeline
## Exact Replication of Joshi et al., Frontiers in Artificial Intelligence, 2025
### doi: 10.3389/frai.2025.1601815

---

## 📋 Paper Summary

**"A nnU-Net-based automatic segmentation of FCD type II lesions in 3D FLAIR MRI images"**  
Joshi et al. (2025), *Frontiers in Artificial Intelligence*, 8:1601815

### Key Results (Paper Table 1)

| Fold | Mean PDS | Final PDS |
|------|----------|-----------|
| 1    | 0.42     | 0.42      |
| 2    | 0.29     | 0.40      |
| 3    | 0.33     | 0.47      |
| 4    | 0.35     | 0.42      |
| 5    | 0.47     | **0.52**  |
| **Overall** | **0.37 ± 0.07** | — |

### Architecture Summary (Exact Paper Settings)

| Setting | Value |
|---------|-------|
| Framework | nnU-Net v2 (3D full-resolution) |
| Input | 3D FLAIR only (single channel) |
| Slice selection | Top-5 axial slices per subject |
| FCD slice ranking | Most lesion/ROI voxels |
| Control slice ranking | Highest FLAIR peak intensity |
| Trainer | nnUNetTrainer_100epochs (custom) |
| Epochs | 100 |
| Optimizer | SGD, momentum=0.99, Nesterov |
| Learning rate | 0.01 (polynomial decay) |
| Loss | Dice + Cross-Entropy |
| Cross-validation | 5-fold |
| Dataset | 85 FCD + 85 controls = 170 cases |
| MC-Dropout | T=20 passes, p=0.2 |

---

## 📁 Project Structure

```
fcd_segmentation/
├── README.md                    ← This file
├── requirements.txt             ← All Python dependencies
├── setup.sh                     ← One-time environment setup script
├── run_pipeline.sh              ← Master script: runs full pipeline end-to-end
│
├── configs/
│   └── config.yaml              ← All hyperparameters and paths (paper settings)
│
├── src/
│   ├── __init__.py              ← Package init + paper result constants
│   ├── utils.py                 ← Utility functions (Dice, PDS formula, etc.)
│   ├── preprocessing.py         ← Resample, skull-strip, normalize, top-K slices
│   ├── dataset.py               ← Subject discovery + nnU-Net JSON builder
│   ├── train.py                 ← nnU-Net training wrapper
│   ├── nnunet_trainer_100ep.py  ← Custom 100-epoch trainer (paper setting)
│   ├── inference.py             ← Ensemble inference + MC-Dropout uncertainty
│   ├── evaluate.py              ← Metrics, stats, Table 1 & 2 replication
│   └── visualize.py             ← Segmentation overlays + training curves
│
├── scripts/
│   ├── 01_download_data.py      ← Download OpenNeuro ds004199
│   ├── 02_preprocess.py         ← Run preprocessing pipeline
│   ├── 03_prepare_nnunet.py     ← Build dataset.json + fingerprint
│   ├── 04_train.py              ← Train nnU-Net (all folds)
│   ├── 05_inference.py          ← Run inference + MC-Dropout
│   ├── 06_evaluate.py           ← Compute metrics + stats
│   └── 07_visualize.py          ← Generate all figures
│
├── outputs/                     ← All generated outputs (auto-created)
│   ├── dataset/                 ← Raw OpenNeuro download (~20 GB)
│   ├── work/
│   │   ├── nnUNet_raw/          ← Preprocessed 5-slice FLAIR volumes
│   │   ├── nnUNet_preprocessed/ ← nnU-Net internal preprocessing
│   │   ├── nnUNet_results/      ← Trained model checkpoints
│   │   ├── skull_stripped/      ← HD-BET skull-stripped outputs
│   │   ├── predictions/         ← Standard nnU-Net predictions
│   │   └── mc_dropout_predictions/ ← MC-Dropout predictions + uncertainty
│   └── figures/
│       ├── segmentation_overlays.png  ← Paper Figure 4
│       └── training_curves.png        ← Paper Figure 5
│
└── tests/
    └── test_utils.py            ← Unit tests for utility functions
```

---

## 🖥️ System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Ubuntu 20.04 / macOS 12 / Windows WSL2 | Ubuntu 22.04 |
| Python | 3.10 | 3.10 |
| RAM | 16 GB | 32 GB |
| GPU VRAM | 8 GB (NVIDIA) | 16–24 GB |
| Disk space | 50 GB | 100 GB |
| CUDA | 11.8 | 12.1 |

> **No GPU?** Set `device: cpu` in `configs/config.yaml`. Training will be ~10× slower.

---

## ⚡ Quick Start (5 steps)

### Step 1 — Set up the project
```bash
chmod +x setup.sh
./setup.sh
```
This creates a conda env named `fcd_seg` and installs everything.

### Step 2 — Activate environment
```bash
conda activate fcd_seg
```

### Step 3 — Run the full pipeline
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

Or run step-by-step (see below).

---

## 🔢 Step-by-Step Execution (Paper Exact Replication)

```bash
conda activate fcd_seg

# Step 1: Download dataset (~30–60 min, ~20 GB)
# Dataset: ds004199, 85 FCD + 85 controls = 170 subjects
python scripts/01_download_data.py

# Step 2: Preprocess ALL subjects (FCD + controls, ~1–3 hrs)
# - 1 mm isotropic resampling
# - HD-BET skull stripping
# - Z-score normalisation
# - Top-5 axial slice selection
#   (FCD: most ROI voxels; Controls: highest FLAIR intensity)
# - Single-channel FLAIR output (no T1!)
python scripts/02_preprocess.py

# Step 3: Build nnU-Net dataset + fingerprint (~5–15 min)
# - Writes single-channel dataset.json
# - Installs custom 100-epoch trainer
# - Runs nnU-Net fingerprinting
python scripts/03_prepare_nnunet.py

# Step 4: Train nnU-Net — 5 folds × 100 epochs (~12–48 hrs)
# - Trainer: nnUNetTrainer_100epochs
# - SGD, LR=0.01, momentum=0.99, Nesterov
# - Loss: Dice + Cross-Entropy
python scripts/04_train.py

# Step 5: Run inference + MC-Dropout uncertainty (~30–60 min)
# - Ensemble: all 5 fold checkpoints
# - MC-Dropout: T=20 passes, p=0.2
python scripts/05_inference.py

# Step 6: Evaluate metrics + statistics
# Outputs: Table 1 (per-fold PDS) + Table 2 (method comparison)
python scripts/06_evaluate.py

# Step 7: Generate all figures
# Outputs: Figure 4 (overlays) + Figure 5 (training curves)
python scripts/07_visualize.py
```

---

## ⚙️ Configuration

All settings live in `configs/config.yaml`. Key paper-exact options:

```yaml
# Paper settings (DO NOT CHANGE for exact replication)
use_subset: false          # Use all 170 subjects (85 FCD + 85 controls)
nnunet:
  config: 3d_fullres       # 3D full-resolution (paper)
  trainer: nnUNetTrainer_100epochs  # Custom 100-epoch trainer
  n_folds: 5               # 5-fold cross-validation
preprocessing:
  top_k_slices: 5          # Top-5 axial slices (paper's novelty)
training:
  epochs: 100              # Paper: 100 epochs
  learning_rate: 0.01      # Paper: LR=0.01
  momentum: 0.99           # Paper: SGD momentum=0.99
mc_dropout:
  n_samples: 20            # Paper: T=20 MC samples
  dropout_p: 0.2           # Paper: p=0.2 dropout
```

---

## 📊 Expected Results (Matching Paper)

| Mode | Mean PDS | Best PDS | Time |
|------|----------|----------|------|
| 10-patient subset, 10 epochs | 0.20–0.35 | ~0.40 | 1–2 hrs |
| **170 patients, 100 epochs (paper)** | **0.37 ± 0.07** | **0.52** | 12–48 hrs |

### Paper Method Comparison (Table 2)

| Method | Dataset | Score |
|--------|---------|-------|
| CNN Encoder-Decoder (House et al., 2021) | 201 FCD | DSC=0.341 |
| MATPR-UNet (Zhang et al., 2024) | EPISURG | DSC=0.42±0.08 |
| NetPos CNN (Feng et al., 2020) | 18 FLAIR-neg | DSC=0.5268 |
| **nnU-Net (Joshi et al., 2025)** | **85 FCD+85 ctrl** | **PDS=0.52 (best)** |

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA out of memory` | Set `batch_size: 2` in config.yaml |
| HD-BET fails | Set `use_skull_strip: false` in config.yaml |
| Download interrupted | Re-run `01_download_data.py` — it resumes |
| `nnUNetTrainer_100epochs not found` | Run step 3 first (it auto-installs), or set `trainer: nnUNetTrainer` |
| Low PDS with subset | Expected — use full 170-patient dataset for paper results |
| nnU-Net fingerprint error | Need ≥ 2 labeled cases; check label paths |
| `ModuleNotFoundError` | Run `./setup.sh` again in the correct conda env |

---

## 📖 References

1. **Joshi et al. (2025)**. *A nnU-Net-based automatic segmentation of FCD type II lesions in 3D FLAIR MRI images.* Front. Artif. Intell. 8:1601815. doi: 10.3389/frai.2025.1601815

2. **Isensee et al. (2021)**. *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.* Nat. Methods 18, 203–211.

3. **Schuch et al. (2023)**. *An open presurgery MRI dataset of people with epilepsy and FCD type II.* Sci. Data 10:475. [OpenNeuro ds004199]