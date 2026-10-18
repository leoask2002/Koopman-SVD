"""Small neural feature maps used by the bounded CPU experiments."""

from __future__ import annotations

import torch
from torch import nn


class MLPFeatures(nn.Module):
    """Tanh multilayer perceptron, or an affine map when hidden_dims is empty.

    An optional constant is handled by the estimator, not this module.  A
    zero-dimensional output supports a constant-only rank-one approximation.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims=(32, 32)):
        super().__init__()
        if input_dim < 1 or output_dim < 0 or any(h < 1 for h in hidden_dims):
            raise ValueError("invalid network dimensions")
        self.output_dim = output_dim
        layers = []
        if output_dim:
            widths = (input_dim, *hidden_dims, output_dim)
            for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
                layers.append(nn.Linear(left, right))
                if index < len(widths) - 2:
                    layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if not self.output_dim:
            return x.new_empty((x.shape[0], 0))
        return self.layers(x)
