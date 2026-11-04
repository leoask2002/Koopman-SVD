"""Finite-rank Koopman composition under changing state distributions.

The columns of a left basis live at the *present* time and the columns of a
right basis at the *next* time.  Adjacent bases need not agree, even when both
are orthonormal; their overlap is estimated on the same intermediate states.
All expectations below use complete training paths.  The caller must reserve
whole other paths for model selection, risk calibration and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from numbers import Integral

import numpy as np


def _matrix(a, label, *, n=None, column=False):
    if np.iscomplexobj(a):
        raise ValueError(f"{label} must be real")
    a = np.asarray(a, dtype=np.float64)
    if column and a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2 or min(a.shape) < 1 or not np.isfinite(a).all() or (n is not None and len(a) != n):
        raise ValueError(f"{label} must be a finite nonempty matrix with the correct number of rows")
    return a


def _integer(a, label, lower):
    if isinstance(a, bool) or not isinstance(a, Integral) or a < lower:
        raise ValueError(f"{label} must be an integer >= {lower}")
    return int(a)


def _sequence(value, length, label, is_single):
    if is_single(value):
        return (value,) * length
    try:
        sequence = tuple(value)
    except TypeError as exc:
        raise ValueError(f"invalid {label}") from exc
    if len(sequence) != length:
        raise ValueError(f"{label} must have one entry per transition")
    return sequence


def _ranks(rank, length):
    return tuple(None if k is None else _integer(k, "rank", 1)
                 for k in _sequence(rank, length, "rank", lambda x: x is None or isinstance(x, Integral)))


def _mean(a):
    # Preserve a constant column exactly even if its value is not exactly
    # representable in binary; naive summation can invent a spurious mode.
    return a[0] + (a - a[0]).mean(axis=0)


def _features(x, fn):
    return _matrix(x if fn is None else fn(x), "feature dictionary", n=len(x), column=True)


def _whiten(raw, rcond):
    center = _mean(raw)
    centered = raw - center
    # An SVD of the centered design avoids squaring its condition number.
    # rcond refers to Gram eigenvalues and hence sqrt(rcond) to singular values.
    _, s, vh = np.linalg.svd(centered, full_matrices=False)
    keep = s > s[0] * max(np.sqrt(rcond), np.finfo(float).eps * max(raw.shape))
    transform = vh[keep].T * (np.sqrt(len(raw)) / s[keep])
    return center, transform, centered @ transform


@dataclass(frozen=True)
class _Basis:
    dictionary: object
    mean: np.ndarray
    transform: np.ndarray

    def __call__(self, x):
        raw = _features(x, self.dictionary)
        if raw.shape[1] != self.mean.size:
            raise ValueError("feature width changed after fit")
        return np.column_stack((np.ones(len(x)), (raw - self.mean) @ self.transform))

    def prefix(self, rank):
        return _Basis(self.dictionary, self.mean, self.transform[:, :rank - 1])


@dataclass(frozen=True)
class SpectralStep:
    """One empirical SVD, with the constant exactly pinned to column zero."""

    left_functions: _Basis
    right_functions: _Basis
    singular_values: np.ndarray
    _target_states: np.ndarray
    _target_basis: np.ndarray

    def coefficients(self, observable):
        """Right-basis coefficients of an observable under this target marginal."""
        x = self._target_states
        values = x if observable is None else _matrix(observable(x), "observable", n=len(x), column=True)
        mean = _mean(values)
        return np.vstack((mean, self._target_basis[:, 1:].T @ (values - mean) / len(x)))

    def prefix(self, rank):
        k = len(self.singular_values) if rank is None else min(rank, len(self.singular_values))
        return SpectralStep(self.left_functions.prefix(k), self.right_functions.prefix(k),
                            self.singular_values[:k], self._target_states, self._target_basis[:, :k])


def _step(x, y, f, g, rank, rcond):
    fm, fw, left = _whiten(_features(x, f), rcond)
    gm, gw, right = _whiten(_features(y, g), rcond)
    # Deflating the constant explicitly avoids arbitrary rotations with other
    # singular values of one, as can occur on deterministic/nonergodic strata.
    u, s, vh = np.linalg.svd(left.T @ right / len(x), full_matrices=False)
    count = len(s) if rank is None else min(rank - 1, len(s))
    l_basis = _Basis(f, fm, fw @ u[:, :count])
    r_basis = _Basis(g, gm, gw @ vh[:count].T)
    return SpectralStep(l_basis, r_basis, np.r_[1., s[:count]], y, r_basis(y))


@dataclass(frozen=True)
class TimeIndexedKoopmanSVD:
    """Two-sided finite-rank model for a finite sequence of transition laws.

    The estimator is the empirical SVD *inside the supplied feature spans*.
    It does not optimize network weights. Use fitted neural maps to measure
    the effect of representation learning; a random neural map is a baseline.
    Singular-value cutoffs/rank selection concern this empirical compression,
    not the global spectrum of the underlying process. Forecasts are real
    observables and need not preserve positivity.
    """

    steps: tuple[SpectralStep, ...]
    bridges: tuple[np.ndarray, ...]
    input_dim: int

    @classmethod
    def fit(cls, train_paths, *, rank=None, left_features=None, right_features=None, rcond=1e-10):
        """Fit from (trajectory, time, state) data, aligned on a common time grid.

        Both dictionary arguments take either a callable of batched states,
        a per-transition sequence of callables, or None for identity features.
        Feature functions must be deterministic and frozen for the life of the
        fit. The constant is always included; any redundant dictionary columns
        (including explicit constants) are removed by Gram truncation. Rank is
        the total cap including the constant; it can vary by transition.
        """
        if np.iscomplexobj(train_paths):
            raise ValueError("train_paths must be real-valued")
        paths = np.array(train_paths, dtype=np.float64, copy=True)
        if paths.ndim != 3 or paths.shape[0] < 1 or paths.shape[1] < 2 or paths.shape[2] < 1 or not np.isfinite(paths).all():
            raise ValueError("train_paths must be a finite array of shape (n, T+1, d), T >= 1")
        if not np.isfinite(rcond) or not 0 < rcond < 1:
            raise ValueError("rcond must be in (0, 1)")
        T = paths.shape[1] - 1
        ranks = _ranks(rank, T)
        is_map = lambda v: v is None or callable(v)
        left = _sequence(left_features, T, "left_features", is_map)
        right = _sequence(right_features, T, "right_features", is_map)
        if not all(is_map(v) for v in (*left, *right)):
            raise ValueError("feature dictionaries must be callables or None")
        steps = tuple(_step(paths[:, t], paths[:, t + 1], left[t], right[t], ranks[t], rcond) for t in range(T))
        bridges = []
        for t in range(T - 1):
            psi = steps[t]._target_basis
            phi = steps[t + 1].left_functions(paths[:, t + 1])
            b = np.zeros((psi.shape[1], phi.shape[1]))
            b[0, 0] = 1.
            b[1:, 1:] = psi[:, 1:].T @ phi[:, 1:] / len(paths)
            bridges.append(b)
        return cls(steps, tuple(bridges), paths.shape[2])

    @classmethod
    def from_neural_models(cls, train_paths, models, *, rank=None, rcond=1e-10):
        """Calibrate already fitted NeuralKoopmanSVD dictionaries per transition.

        The models are not retrained or modified. Calibration uses only
        train_paths, with the pinned constant and rank policy of this class.
        Supply one model per transition (repeat a pooled model for a shared
        dictionary) and do not mutate those models afterward.
        """
        models = tuple(models)
        if not all(callable(getattr(m, "raw_functions", None)) for m in models):
            raise TypeError("each model must expose a fitted raw_functions method")
        return cls.fit(train_paths, rank=rank, rcond=rcond,
                       left_features=[partial(m.raw_functions, side="left") for m in models],
                       right_features=[partial(m.raw_functions, side="right") for m in models])

    def truncate(self, rank):
        """Retain nested prefixes of an existing fit (never restores dropped modes)."""
        steps = tuple(step.prefix(k) for step, k in zip(self.steps, _ranks(rank, len(self.steps))))
        bridges = tuple(b[:len(steps[t].singular_values), :len(steps[t+1].singular_values)]
                        for t, b in enumerate(self.bridges))
        return TimeIndexedKoopmanSVD(steps, bridges, self.input_dim)

    def predict(self, x, *, start=0, horizon=1, observable=None):
        """Predict E[observable(X[start+horizon]) | X[start]=x], as an (n,m) array.

        The decoder uses the terminal training marginal and bridges use only
        aligned intermediate training states. No held-out data or recursively
        encoded predicted states appear in those moments.
        """
        start, horizon = _integer(start, "start", 0), _integer(horizon, "horizon", 0)
        if start + horizon > len(self.steps):
            raise ValueError("forecast extends beyond the fitted time grid")
        x = _matrix(x, "states")
        if x.shape[1] != self.input_dim:
            raise ValueError("state dimension does not match training paths")
        if horizon == 0:
            return x.copy() if observable is None else _matrix(observable(x), "observable", n=len(x), column=True)
        end = start + horizon - 1
        last = self.steps[end]
        coeff = last.singular_values[:, None] * last.coefficients(observable)
        for t in range(end - 1, start - 1, -1):
            coeff = self.steps[t].singular_values[:, None] * (self.bridges[t] @ coeff)
        prediction = self.steps[start].left_functions(x) @ coeff
        if not np.isfinite(prediction).all():
            raise FloatingPointError("projection produced nonfinite predictions")
        return prediction
