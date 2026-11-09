#!/usr/bin/env python3
"""
FastAD Smoke Test Suite
-----------------------
Lightweight CPU-compatible verification test for FastAD pipeline components.
Runs in seconds without needing GPUs or external datasets.

Usage:
    python tests/test_smoke.py
"""

import sys
import unittest
from pathlib import Path
import torch
import numpy as np

# Ensure fastad package is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fastad.feature_extractor import FeatureExtractor
from unittest.mock import patch
from fastad.patch_extractor import PatchExtractor
from fastad.set_features import compute_cdf_and_binary_features
from fastad.gaussian_estimator import GaussianEstimator
from fastad.config_utils import load_config


class TestFastADComponents(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.device = torch.device("cpu")

    def test_backbone(self):
        """Exercise the supported timm path without downloading weights."""
        from timm import create_model

        backbone = create_model("resnet18", pretrained=False, features_only=True)
        with patch("fastad.feature_extractor.timm.create_model", return_value=backbone) as create:
            model = FeatureExtractor("resnet18", layer_to_extract=2, device=self.device)
        create.assert_called_once_with("resnet18", pretrained=True, features_only=True)
        self.assertFalse(backbone.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in backbone.parameters()))
        with torch.no_grad():
            out = model(torch.randn(2, 3, 64, 64, device=self.device))
        self.assertEqual(out.shape[:2], (2, 128))

    def test_patch_extractor(self):
        """Test PatchExtractor unfolding with same-padding."""
        extractor = PatchExtractor(patchsize=3, stride=1)
        feature_map = torch.randn(2, 64, 32, 32)
        patches, num_patches = extractor(feature_map, return_spatial_info=True)
        # Spatial size preserved with padding=1: 32x32 = 1024 patches per image
        self.assertEqual(num_patches, [32, 32])
        self.assertEqual(patches.dim(), 3)
        self.assertEqual(patches.shape[0], 2 * 32 * 32)
        self.assertEqual(patches.shape[1], 64)
        self.assertEqual(patches.shape[2], 9)
        print("[PASS] PatchExtractor passed.")

    def test_cdf_set_features(self):
        """Test both quantile modes on fixed CPU features."""
        # [B*N, C, P*P]
        patches = torch.arange(270, device=self.device, dtype=torch.float32).reshape(10, 3, 9)
        n_quantiles = 3
        outputs = []
        for method in ("channelwise", "vectorized"):
            with self.subTest(quantile_method=method):
                cdf_features = compute_cdf_and_binary_features(
                    patches,
                    n_quantiles=n_quantiles,
                    min_q=0.1,
                    max_q=0.9,
                    quantile_method=method,
                )
                self.assertEqual(cdf_features.shape, (10, 3 * n_quantiles))
                self.assertEqual(cdf_features.device, patches.device)
                self.assertTrue(((cdf_features >= 0.0) & (cdf_features <= 1.0)).all())
                outputs.append(cdf_features)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        default_features = compute_cdf_and_binary_features(
            patches, n_quantiles=n_quantiles, min_q=0.1, max_q=0.9
        )
        torch.testing.assert_close(default_features, outputs[0], rtol=0, atol=0)
        print("[PASS] CDF feature extraction passed.")

    def test_gaussian_estimator(self):
        """Test Gaussian distribution fitting and Mahalanobis scoring."""
        # Synthetic features: 50 samples, 8 dimensions
        features = torch.randn(50, 8, device=self.device)
        estimator = GaussianEstimator(covariance_method="LedoitWolf")
        estimator.fit(features)
        
        self.assertEqual(estimator.mean.shape, (8,))
        self.assertEqual(estimator.inv_cov.shape, (8, 8))
        
        # Test Mahalanobis distance calculation
        test_features = torch.randn(5, 8, device=self.device)
        dist = estimator.compute_distance(test_features)
        self.assertEqual(dist.shape[0], 5)
        self.assertTrue((dist >= 0).all())
        print("[PASS] GaussianEstimator and Mahalanobis distance passed.")

    def test_config_resolution(self):
        """Test OmegaConf paths.yaml interpolation."""
        cfg_path = REPO_ROOT / "configs" / "MVTec_AD" / "bottle" / "base.yaml"
        if cfg_path.exists():
            config = load_config(str(cfg_path))
            self.assertIn("dataset", config)
            self.assertIn("model", config)
            self.assertIn("paths", config)
            self.assertTrue(isinstance(config["dataset"]["source"], str))
            # Verify source resolved to ./data/mvtec
            self.assertTrue("mvtec" in config["dataset"]["source"])
            print("[PASS] Config resolution and variable interpolation passed.")


if __name__ == "__main__":
    print("=" * 60)
    print("Running FastAD Smoke Tests...")
    print("=" * 60)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFastADComponents)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\nALL FASTAD SMOKE TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("\nSmoke tests failed.")
        sys.exit(1)
