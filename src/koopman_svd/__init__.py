"""Modern, independently usable neural Koopman SVD package."""

from .losses import nesting_masks, neural_svd_loss
from .model import NeuralKoopmanSVD
from .networks import MLPFeatures

__all__ = ["NeuralKoopmanSVD", "MLPFeatures", "nesting_masks", "neural_svd_loss"]
__version__ = "0.2.0"
