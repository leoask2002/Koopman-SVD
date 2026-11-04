"""Finite-menu risk bounds for scalar, bounded conditional-mean predictions.

See docs/certification.md for the sampling contract and proof. Each start is one
independent unit; correlated horizons within that unit do not increase sample size.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class RiskCertificate:
    """One-sided simultaneous bounds in squared observable units, by model/horizon."""

    estimate: np.ndarray  # signed cross-product mean, or noisy MSE in single mode
    upper: np.ndarray
    n_starts: int
    delta: float
    bounds: tuple[float, float]
    paired: bool
    method: str


def _samples(predictions, outcomes, bounds, delta):
    p = np.asarray(predictions, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if p.ndim != 3 or min(p.shape) < 1 or y.shape != p.shape[1:]:
        raise ValueError("expected (models, horizons, starts) and (horizons, starts)")
    if p.shape[-1] < 2:
        raise ValueError("at least two independent starts are required")
    a, b = map(float, bounds)
    width = b - a
    if not (0.0 < width and np.isfinite(width * width) and width * width > 0):
        raise ValueError("bounds must be increasing and squared range representable")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1)")
    if not np.isfinite(p).all():
        raise ValueError("predictions must be finite")
    if not np.isfinite(y).all() or np.any((y < a) | (y > b)):
        raise ValueError("outcomes lie outside declared finite observable bounds")
    return np.clip(p, a, b), y, a, b


def _radius(samples, *, diameter, count, delta, method):
    # samples are in units of range squared, so their variance cannot overflow.
    n = samples.shape[-1]
    log_menu_delta = np.log(count) - np.log(delta)
    if method == "empirical_bernstein":
        t = log_menu_delta + np.log(2.0)
        variance = np.var(samples, axis=-1, ddof=1)
        return np.sqrt(2 * variance * t / n) + 7 * diameter * t / (3 * (n - 1))
    if method == "hoeffding":
        return diameter * np.sqrt(log_menu_delta / (2 * n))
    raise ValueError("method must be empirical_bernstein or hoeffding")


def certify_forecasts(
    predictions,
    outcomes,
    replicas=None,
    *,
    bounds: tuple[float, float],
    delta: float = 0.05,
    method: Literal["empirical_bernstein", "hoeffding"] = "empirical_bernstein",
) -> RiskCertificate:
    """Certify the risk of *clipped* predictions over a predeclared finite menu.

    predictions: (candidate, horizon, independent_start), possibly outside bounds.
    outcomes and optional replicas: (horizon, independent_start) within the true
    observable bounds. Replicas must be independent continuations conditional on
    each common start. Without replicas the upper bound also includes outcome noise.
    All model choices, calibration of dictionaries/bridges, and the clipping range
    must be fixed independently of these starts. Deploy the same clipping rule.
    """
    p, y, a, b = _samples(predictions, outcomes, bounds, delta)
    width = b - a
    z = y if replicas is None else np.asarray(replicas, dtype=float)
    if z.shape != y.shape:
        raise ValueError("replicas must have the same shape as outcomes")
    if not np.isfinite(z).all() or np.any((z < a) | (z > b)):
        raise ValueError("replicas lie outside declared finite observable bounds")

    losses = ((p - y) / width) * ((p - z) / width)
    estimate = losses.mean(axis=-1)
    diameter = 1.0 if replicas is None else 1.25
    radius = _radius(losses, diameter=diameter, count=estimate.size, delta=delta, method=method)
    return RiskCertificate(
        estimate=estimate * width**2,
        upper=np.clip(estimate + radius, 0.0, 1.0) * width**2,
        n_starts=p.shape[-1], delta=float(delta), bounds=(a, b), paired=replicas is not None,
        method=method,
    )


@dataclass(frozen=True)
class ImprovementCertificate:
    """Positive lower bound certifies strictly lower risk than the stated baseline."""

    estimate: np.ndarray
    lower: np.ndarray
    baseline: int
    n_starts: int
    delta: float
    bounds: tuple[float, float]
    method: str


def _finite_support_diameter(support, indices, p, a, b, baseline):
    """Normalized gain diameters on an externally known finite start support."""
    if np.iscomplexobj(support):
        raise ValueError("support_predictions must be real")
    support = np.asarray(support, dtype=float)
    if (support.ndim != 3 or support.shape[:2] != p.shape[:2] or support.shape[2] < 1
            or not np.isfinite(support).all()):
        raise ValueError("support_predictions must be finite (models, horizons, support points)")
    ids = np.asarray(indices)
    if ids.shape == (p.shape[-1],):
        ids = np.broadcast_to(ids, p.shape[1:])
    if (not np.issubdtype(ids.dtype, np.integer) or ids.shape != p.shape[1:] or
            np.any(ids < 0) or np.any(ids >= support.shape[-1])):
        raise ValueError("support_indices must index every calibration start on the supplied support")
    support = np.clip(support, a, b)
    observed = np.take_along_axis(support, ids[None], axis=-1)
    if not np.array_equal(observed, p):
        raise ValueError("clipped calibration predictions do not match the frozen support table")

    # Squared-loss differences are affine in y, so interval endpoints suffice.
    v = (support - a) / (b-a)
    baseline_v = v[baseline]
    at_zero = baseline_v**2 - v**2
    at_one = (baseline_v - 1)**2 - (v - 1)**2
    lo = np.minimum(at_zero, at_one).min(axis=-1)
    hi = np.maximum(at_zero, at_one).max(axis=-1)
    # Outward round a nonzero span; the usual global bound still always applies.
    return np.where(hi == lo, 0., np.minimum(2., np.nextafter(hi-lo, np.inf)))


def certify_improvement(
    predictions, outcomes, *, baseline: int, bounds: tuple[float, float],
    delta: float = 0.05,
    support_predictions=None,
    support_indices=None,
    method: Literal["empirical_bernstein", "hoeffding"] = "empirical_bernstein",
) -> ImprovementCertificate:
    """Bound baseline minus candidate conditional-mean MSE with one outcome/start.

    Same sampling, clipping and shapes as certify_forecasts. The baseline index
    must be chosen before calibration. Noise variance cancels in the difference
    of losses against the same outcome. Bounds are simultaneous within this call.

    Optional finite-support refinement: provide the frozen predictor table
    (models, horizons, support points) on the independently known COMPLETE
    support of starts, and support_indices of shape (starts,) or (horizons,
    starts). Calibration predictions must equal the corresponding clipped table
    entries. The caller must establish completeness; sampled extrema are invalid.
    """
    p, y, a, b = _samples(predictions, outcomes, bounds, delta)
    if isinstance(baseline, bool) or not isinstance(baseline, (int, np.integer)):
        raise ValueError("baseline must index a candidate")
    if len(p) < 2 or not 0 <= baseline < len(p):
        raise ValueError("at least two candidates and a valid baseline index are required")
    width = b - a
    errors = ((p - y) / width) ** 2
    gain = errors[baseline] - errors
    mean = gain.mean(axis=-1)
    count = (len(p) - 1) * p.shape[1]
    if (support_predictions is None) != (support_indices is None):
        raise ValueError("provide both support_predictions and support_indices, or neither")
    diameter = (2.0 if support_predictions is None else
                _finite_support_diameter(support_predictions, support_indices, p, a, b, baseline))
    radius = _radius(gain, diameter=diameter, count=count, delta=delta, method=method)
    lower = np.maximum(-1.0, mean - radius) * width**2
    lower[baseline] = 0.0  # the baseline compared with itself is identically zero
    return ImprovementCertificate(
        estimate=mean * width**2, lower=lower, baseline=int(baseline), n_starts=p.shape[-1],
        delta=float(delta), bounds=(a, b), method=method,
    )


@dataclass(frozen=True)
class Selection:
    """Selected menu index, rank, and length of initial acceptable horizon prefix."""

    index: int | None
    rank: int | None
    horizon: int


def select_horizon(certificate: RiskCertificate, *, ranks, tolerance: float) -> Selection:
    """Maximize certified prefix at the supplied absolute RMSE tolerance.

    Ties prefer smaller rank, then earlier menu index. Horizon zero abstains. For
    nonconsecutive target times, horizon means number of checked prefix entries;
    it makes no claim about physically intermediate times.
    """
    upper = certificate.upper
    ranks = np.asarray(ranks)
    if ranks.shape != (upper.shape[0],) or not np.issubdtype(ranks.dtype, np.integer):
        raise ValueError("ranks must be one positive integer per candidate")
    if np.any(ranks < 1):
        raise ValueError("ranks must be positive")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite nonnegative RMSE")
    prefix = np.logical_and.accumulate(np.sqrt(upper) <= tolerance, axis=1).sum(axis=1)
    longest = int(prefix.max())
    if longest == 0:
        return Selection(None, None, 0)
    eligible = np.flatnonzero(prefix == longest)
    best = int(eligible[np.argmin(ranks[eligible])])
    return Selection(best, int(ranks[best]), longest)
