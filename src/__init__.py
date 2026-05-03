# FCD Segmentation Pipeline — src package
# Based on Joshi et al., Frontiers in Artificial Intelligence, 2025
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH  = PROJECT_ROOT / "configs" / "config.yaml"

# Paper details
PAPER_TITLE = "A nnU-Net-based automatic segmentation of FCD type II lesions in 3D FLAIR MRI"
PAPER_DOI   = "10.3389/frai.2025.1601815"
PAPER_RESULTS = {
    "mean_pds": 0.37,
    "std_pds":  0.07,
    "best_pds": 0.52,  # Fold 5
    "fold_results": {   # Table 1: (mean_pds, final_pds)
        1: (0.42, 0.42),
        2: (0.29, 0.40),
        3: (0.33, 0.47),
        4: (0.35, 0.42),
        5: (0.47, 0.52),
    }
}