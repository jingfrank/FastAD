"""FastAD: A Deployable Framework for Real-Time Visual Anomaly Detection."""


from .fastad import FastAD
from .feature_extractor import FeatureExtractor
from .gaussian_estimator import GaussianEstimator
from .patch_extractor import PatchExtractor

__version__ = "1.0.0"
__all__ = ["FastAD", "FeatureExtractor", "GaussianEstimator", "PatchExtractor"]
