"""Deterministic quantile-mode regression tests; no datasets or weight downloads."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fastad.fastad import FastAD
from fastad.set_features import compute_cdf_and_binary_features


METHODS = ("channelwise", "vectorized")


def scalar_quantile_reference(features, n_quantiles, min_q, max_q):
    """Preserve the original scalar-quantile thresholds and output ordering."""
    work = features.float() if features.dtype == torch.float16 else features
    channel_cdfs = []
    for channel in range(work.shape[1]):
        values = work[:, channel, :]
        lower = torch.quantile(values.reshape(-1), min_q)
        upper = torch.quantile(values.reshape(-1), max_q)
        bins = []
        for index in range(n_quantiles):
            threshold = lower + (upper - lower) * (index + 1) / (n_quantiles + 1)
            bins.append((values >= threshold).float().mean(dim=-1))
        channel_cdfs.append(torch.stack(bins, dim=1))
    # [N, bins, C] keeps the existing threshold-major flattening order.
    return torch.stack(channel_cdfs, dim=2).reshape(features.shape[0], -1)


class TestQuantileMethods(unittest.TestCase):
    def test_modes_match_scalar_reference(self):
        varied = torch.tensor([
            [[-8, -3, 0, 2, 9], [4, 4, 5, 7, 12], [30, -2, 6, 1, 10]],
            [[7, -1, 3, 3, 20], [-5, 0, 4, 8, 8], [2, 11, -9, 5, 5]],
        ])
        cases = {
            "varied": varied,
            "constant": torch.full((2, 3, 5), 2.0),
            "single_channel": varied[:, :1, :],
            "single_sample": varied[:1],
            "single_spatial_value": varied[:, :, :1],
            "singleton": torch.tensor([[[3.0]]]),
            "noncontiguous": varied.transpose(0, 1),
        }
        for dtype in (torch.float32, torch.float64, torch.float16):
            for name, values in cases.items():
                features = values.to(dtype=dtype)
                for n_quantiles, min_q, max_q in ((1, 0.0, 1.0), (3, 0.25, 0.75), (5, 0.1, 0.9)):
                    expected = scalar_quantile_reference(features, n_quantiles, min_q, max_q)
                    for method in METHODS:
                        with self.subTest(dtype=dtype, case=name, bins=n_quantiles, method=method):
                            actual = compute_cdf_and_binary_features(
                                features,
                                n_quantiles=n_quantiles,
                                min_q=min_q,
                                max_q=max_q,
                                quantile_method=method,
                            )
                            self.assertEqual(actual.shape, (features.shape[0], features.shape[1] * n_quantiles))
                            self.assertEqual(actual.device, features.device)
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_constant_values_include_threshold_equality(self):
        features = torch.full((2, 1, 3), 4.0)
        for method in METHODS:
            with self.subTest(method=method):
                actual = compute_cdf_and_binary_features(features, n_quantiles=3, quantile_method=method)
                torch.testing.assert_close(actual, torch.ones(2, 3), rtol=0, atol=0)

    def test_invalid_method(self):
        features = torch.tensor([[[0.0, 1.0, 2.0]]])
        for method in ("invalid", "", "Channelwise"):
            with self.subTest(method=method):
                with self.assertRaises(ValueError):
                    compute_cdf_and_binary_features(features, quantile_method=method)


class TestFastADQuantileConfig(unittest.TestCase):
    def setUp(self):
        # Only configuration and empty estimator buffers are exercised here.
        extractor = patch("fastad.fastad.FeatureExtractor", return_value=torch.nn.Identity())
        self.extractor = extractor.start()
        self.addCleanup(extractor.stop)
        self.config = {
            "backbone": "resnet18",
            "layer_to_extract": [1],
            "device": torch.device("cpu"),
            "patchsize": 3,
            "stride": 1,
            "target_size": 8,
            "n_quantiles": 3,
            "fm_quantiles": 3,
            "min_q": 0.1,
            "max_q": 0.9,
            "fm_min_q": 0.1,
            "fm_max_q": 0.9,
            "patch_covariance_method": "LedoitWolf",
            "global_covariance_method": "LedoitWolf",
            "regularization": 1e-6,
            "pretrained_weights_path": None,
        }

    def saved_checkpoint(self, model):
        with patch("fastad.fastad.torch.save") as save:
            model.save_model("unused-checkpoint.pth")
        save.assert_called_once()
        return save.call_args.args[0]

    def test_constructor_default(self):
        self.assertEqual(FastAD(**self.config).quantile_method, "channelwise")

    def test_mode_is_saved_and_restored(self):
        for method in METHODS:
            with self.subTest(method=method):
                model = FastAD(**self.config, quantile_method=method)
                self.assertEqual(model.quantile_method, method)
                checkpoint = self.saved_checkpoint(model)
                self.assertEqual(checkpoint["model_config"]["quantile_method"], method)
                with patch("fastad.fastad.torch.load", return_value=checkpoint):
                    restored = FastAD.load_model("unused-checkpoint.pth", device=torch.device("cpu"))
                self.assertEqual(restored.quantile_method, method)

    def test_missing_checkpoint_field_defaults_to_channelwise(self):
        model = FastAD(**self.config, quantile_method="vectorized")
        checkpoint = self.saved_checkpoint(model)
        del checkpoint["model_config"]["quantile_method"]
        with patch("fastad.fastad.torch.load", return_value=checkpoint):
            restored = FastAD.load_model("unused-checkpoint.pth", device=torch.device("cpu"))
        self.assertEqual(restored.quantile_method, "channelwise")

    def test_constructor_rejects_invalid_method(self):
        with self.assertRaises(ValueError):
            FastAD(**self.config, quantile_method="invalid")


if __name__ == "__main__":
    unittest.main()
