import numpy as np
import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import dice_score, pseudo_dice_score, sensitivity, specificity

class TestUtils(unittest.TestCase):
    def test_dice_score(self):
        # Perfect match
        a = np.ones((10, 10))
        self.assertEqual(dice_score(a, a), 1.0)
        
        # Half match
        b = np.zeros((10, 10))
        b[:5, :] = 1
        self.assertAlmostEqual(dice_score(a, b), 2 * 50 / (100 + 50))
        
        # Both empty
        c = np.zeros((10, 10))
        self.assertEqual(dice_score(c, c), 1.0)
        
        # One empty
        self.assertEqual(dice_score(a, c), 0.0)

    def test_pseudo_dice_score(self):
        # Both empty
        c = np.zeros((10, 10))
        # PDS = (0 + eps) / (0 + eps) = 1.0
        self.assertEqual(pseudo_dice_score(c, c), 1.0)
        
        # Near empty
        a = np.zeros((10, 10))
        a[0, 0] = 1
        # PDS = (0 + eps) / (1 + eps)
        self.assertLess(pseudo_dice_score(a, c), 0.01)

    def test_sensitivity(self):
        gt = np.zeros((10, 10))
        gt[:5, :] = 1
        pred = np.zeros((10, 10))
        pred[:3, :] = 1
        # TP = 30, FN = 20, TP+FN = 50. Sens = 30/50 = 0.6
        self.assertEqual(sensitivity(pred, gt), 0.6)
        
        # Empty GT
        c = np.zeros((10, 10))
        self.assertEqual(sensitivity(pred, c), 1.0)

    def test_specificity(self):
        gt = np.zeros((10, 10))
        gt[:5, :] = 1
        pred = np.zeros((10, 10))
        pred[5:7, :] = 1
        # TN = 30, FP = 20, TN+FP = 50. Spec = 30/50 = 0.6
        self.assertEqual(specificity(pred, gt), 0.6)
        
        # All GT positive
        all_pos = np.ones((10, 10))
        self.assertEqual(specificity(pred, all_pos), 1.0)

if __name__ == "__main__":
    unittest.main()
