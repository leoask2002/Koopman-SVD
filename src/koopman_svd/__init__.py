"""Modern, independently usable neural Koopman SVD package."""

from .losses import nesting_masks, neural_svd_loss
from .model import NeuralKoopmanSVD
from .networks import MLPFeatures
from .transport import TimeIndexedKoopmanSVD
from .certification import certify_forecasts, certify_improvement, select_horizon

__all__ = ["NeuralKoopmanSVD", "MLPFeatures", "nesting_masks", "neural_svd_loss",
           "TimeIndexedKoopmanSVD", "certify_forecasts", "certify_improvement", "select_horizon"]
__version__ = "0.3.0"
