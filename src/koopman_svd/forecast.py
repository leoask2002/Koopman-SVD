"""Optional EDMD forecasting from a fitted NeuralSVD feature dictionary.

This module fits a *separate* least-squares forecasting operator. It neither
changes the neural singular functions nor improves their SVD objective. The
dictionary augments calibrated left functions with the observed state and a
constant, so forecasts need not discard state information through an imperfect
right-function decoder. All dictionary normalization and regression use only
the pairs passed to ``fit``.
"""

from __future__ import annotations

import math
import operator

import numpy as np
import torch


class StateAugmentedEDMD:
    """Post-fit EDMD head using ``[1, state, calibrated left functions]``.

    Parameters
    ----------
    model : NeuralKoopmanSVD
        Already fitted model, used only to evaluate its calibrated left
        functions. The caller must refit this head after refitting the model.
    rank : int or None
        Retained training dictionary design-matrix rank. ``None`` retains all
        supported directions according to ``rcond``. This is an EDMD dictionary
        rank, not the NeuralSVD rank.
    ridge : float
        Nonnegative regularizer for mean-squared feature regression; the linear
        solve uses ``s / (s**2 + n_train * ridge)``. Zero is ordinary least
        squares. Ridge also regularizes the state decoder.
    rcond : float
        Relative cutoff on training design-matrix singular values. Redundant
        constants and other unsupported directions are discarded explicitly.

    Notes
    -----
    Multi-step predictions apply powers of a fixed reduced EDMD matrix. They
    do not repeatedly encode predicted states. No stability constraint is
    imposed: inspect ``spectral_diagnostics()`` and validate the desired
    forecast horizons. ``predict(..., steps=0)`` returns the dictionary's state
    reconstruction, which need not be exact after rank truncation or ridge.
    NumPy inputs give float64 NumPy outputs. Floating torch inputs give tensors
    on the same device and with the same dtype; this head is not differentiable.
    """

    def __init__(self, model, *, rank=None, ridge=0.0, rcond=1e-10):
        if not callable(getattr(model, "left_functions", None)):
            raise TypeError("model must expose calibrated left_functions")
        if rank is not None and (isinstance(rank, bool) or not isinstance(rank, int) or rank < 1):
            raise ValueError("rank must be a positive integer or None")
        if not math.isfinite(ridge) or ridge < 0:
            raise ValueError("ridge must be finite and nonnegative")
        if not math.isfinite(rcond) or not 0 < rcond < 1:
            raise ValueError("rcond must be finite and in (0, 1)")
        self.model, self.rank, self.ridge, self.rcond = model, rank, float(ridge), float(rcond)
        self._fitted = False

    @staticmethod
    def _array(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if np.iscomplexobj(value):
            raise ValueError("complex-valued states are not supported by this real-valued forecasting head")
        try:
            result = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("states must be a finite numeric matrix") from exc
        if result.ndim != 2 or min(result.shape) < 1 or not np.isfinite(result).all():
            raise ValueError("states must be a finite nonempty two-dimensional matrix")
        return result

    def _raw(self, x):
        left = self.model.left_functions(x)
        if isinstance(left, torch.Tensor):
            left = left.detach().cpu().numpy()
        left = np.asarray(left, dtype=np.float64)
        if left.ndim != 2 or len(left) != len(x) or not np.isfinite(left).all():
            raise ValueError("model.left_functions must return finite features preserving sample count")
        return np.column_stack((x, left))

    def _features(self, x):
        return np.column_stack((np.ones(len(x)), (self._raw(x) - self.mean_) / self.scale_))

    def fit(self, x, y):
        """Fit only on supplied training pairs; never modifies the neural model."""
        x, y = self._array(x), self._array(y)
        if x.shape != y.shape:
            raise ValueError("x and y must have equal sample counts and state dimensions")
        self._fitted = False
        self.input_dim_ = x.shape[1]
        raw = self._raw(x)
        self.mean_, self.scale_ = raw.mean(axis=0), raw.std(axis=0)
        self.scale_ = np.where(self.scale_ > 1e-12, self.scale_, 1.0)
        px, py = self._features(x), self._features(y)
        u, singular, vh = np.linalg.svd(px, full_matrices=False)
        self.design_singular_values_ = singular
        self.supported_rank_ = int(np.count_nonzero(singular > singular[0] * self.rcond))
        rank = self.supported_rank_ if self.rank is None else self.rank
        if rank > self.supported_rank_:
            raise ValueError(f"rank exceeds supported training dictionary rank {self.supported_rank_}")
        self.effective_rank_ = rank
        self.projection_ = vh[:rank].T
        inverse = (singular[:rank] / (singular[:rank] ** 2 + len(x) * self.ridge))[:, None] * u[:, :rank].T
        self.operator_ = inverse @ (py @ self.projection_)
        self.decoder_ = inverse @ x
        self.training_pairs_ = len(x)
        self._fitted = True
        return self

    def predict(self, x, *, steps=1):
        """Return a finite-rank EDMD state forecast at the requested horizon."""
        if not self._fitted:
            raise RuntimeError("fit the forecasting head before prediction")
        if isinstance(steps, bool):
            raise ValueError("steps must be a nonnegative integer")
        try:
            steps = operator.index(steps)
        except TypeError as exc:
            raise ValueError("steps must be a nonnegative integer") from exc
        if steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        values = self._array(x)
        if values.shape[1] != self.input_dim_:
            raise ValueError(f"states must have {self.input_dim_} columns")
        with np.errstate(over="ignore", invalid="ignore"):
            prediction = self._features(values) @ self.projection_ @ np.linalg.matrix_power(self.operator_, steps) @ self.decoder_
        if not np.isfinite(prediction).all():
            raise FloatingPointError("EDMD forecast overflowed; inspect the spectral radius and reduce the horizon")
        if isinstance(x, torch.Tensor):
            dtype = x.dtype if x.is_floating_point() else torch.float64
            return torch.as_tensor(prediction, dtype=dtype, device=x.device)
        return prediction

    def spectral_diagnostics(self):
        """Return EDMD eigenvalues and conditioning, not Koopman singular values."""
        if not self._fitted:
            raise RuntimeError("fit the forecasting head before requesting diagnostics")
        eigenvalues = np.linalg.eigvals(self.operator_)
        singular = self.design_singular_values_[: self.effective_rank_]
        return {
            "eigenvalues": eigenvalues,
            "spectral_radius": float(np.max(np.abs(eigenvalues))),
            "effective_rank": self.effective_rank_,
            "supported_rank": self.supported_rank_,
            "design_condition_number": float(singular[0] / singular[-1]),
            "ridge": self.ridge,
            "rcond": self.rcond,
            "training_pairs": self.training_pairs_,
        }
