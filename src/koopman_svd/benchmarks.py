"""Small reproducible synthetic systems with transparent reference dynamics.

All generators return NumPy float64 arrays of shape
``(n_trajectories, n_steps + 1, state_dimension)``. Trajectories use independent
initial draws. The nonlinear triangular system is deliberately nonstationary;
its polynomial eigenvalues must not be confused with distribution-dependent
Koopman singular values.
"""

from __future__ import annotations

import numpy as np

from .data import _positive_int


def gaussian_ar1_trajectories(
    n_trajectories: int = 100,
    n_steps: int = 10,
    *,
    seed: int = 0,
    matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Stationary Gaussian ``x[t+1] = A @ x[t] + epsilon`` in column convention.

    Initial states have covariance I and innovations have covariance
    ``I - A @ A.T``. The default A is nonsymmetric with distinct singular values
    0.85, 0.55, 0.25. On the linear-observable subspace, the Koopman singular
    values equal those of A; the full Gaussian operator also has a constant
    singular value 1 and higher-order Hermite modes. Do not compare a generic
    rank-r nonlinear model to only the three linear singular values.

    Returns ``(trajectories, A)``. Row-wise observations satisfy
    ``E[y | x] = x @ A.T`` and ``E[x.T @ y] / n = A.T`` under stationarity.
    """
    n_trajectories = _positive_int(n_trajectories, "n_trajectories")
    n_steps = _positive_int(n_steps, "n_steps")
    if matrix is None:
        theta, phi = 0.43, -0.71
        left = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        right = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
        matrix = left @ np.diag([0.85, 0.55, 0.25]) @ right.T
    a = np.array(matrix, dtype=np.float64, copy=True)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
        raise ValueError("matrix must be a nonempty square array")
    if not np.isfinite(a).all() or np.linalg.norm(a, ord=2) >= 1:
        raise ValueError("matrix must be finite with spectral norm strictly less than 1")
    noise_covariance = np.eye(a.shape[0]) - a @ a.T
    noise_factor = np.linalg.cholesky(noise_covariance)
    rng = np.random.default_rng(seed)
    values = np.empty((n_trajectories, n_steps + 1, a.shape[0]), dtype=np.float64)
    values[:, 0] = rng.normal(size=(n_trajectories, a.shape[0]))
    for t in range(n_steps):
        innovations = rng.normal(size=(n_trajectories, a.shape[0])) @ noise_factor.T
        values[:, t + 1] = values[:, t] @ a.T + innovations
    return values, a


def triangular_trajectories(
    n_trajectories: int = 100,
    n_steps: int = 10,
    *,
    seed: int = 0,
    a: float = 0.8,
    b: float = 0.6,
) -> np.ndarray:
    """Nonlinear map ``(x1,x2) -> (a*x1, b*x2 + (a*a-b)*x1*x1)``.

    Initial states are uniform on [-1, 1]^2. The exact three-dimensional
    invariant dictionary is :func:`triangular_features`; its Koopman eigenvalues
    are a, b, a^2. These are eigenvalues, not general singular-value references.
    """
    n_trajectories = _positive_int(n_trajectories, "n_trajectories")
    n_steps = _positive_int(n_steps, "n_steps")
    if not np.isfinite([a, b]).all() or not (abs(a) < 1 and abs(b) < 1):
        raise ValueError("a and b must be finite and have magnitude less than 1")
    rng = np.random.default_rng(seed)
    values = np.empty((n_trajectories, n_steps + 1, 2), dtype=np.float64)
    values[:, 0] = rng.uniform(-1, 1, size=(n_trajectories, 2))
    for t in range(n_steps):
        values[:, t + 1] = triangular_step(values[:, t], a=a, b=b)
    return values


def triangular_step(x: np.ndarray, *, a: float = 0.8, b: float = 0.6) -> np.ndarray:
    """Apply the triangular map to observations with final dimension two."""
    x = np.asarray(x)
    if x.ndim < 1 or x.shape[-1] != 2:
        raise ValueError("triangular observations must have final dimension 2")
    return np.stack((a * x[..., 0], b * x[..., 1] + (a * a - b) * x[..., 0] ** 2), axis=-1)


def triangular_features(x: np.ndarray) -> np.ndarray:
    """Return the exact dictionary ``[x1, x2, x1^2]`` for NumPy observations."""
    x = np.asarray(x)
    if x.ndim < 1 or x.shape[-1] != 2:
        raise ValueError("triangular observations must have final dimension 2")
    return np.stack((x[..., 0], x[..., 1], x[..., 0] ** 2), axis=-1)


def logistic_trajectories(
    n_trajectories: int = 100,
    n_steps: int = 10,
    *,
    seed: int = 0,
    r: float = 4.0,
) -> np.ndarray:
    """Logistic map, with default r=4 and Beta(1/2,1/2) initial observations.

    The initialization is stationary for r=4 only; other r use the same initial
    distribution without any stationarity claim.
    Returns independent trajectories, not pieces split from one long orbit.
    Finite precision eventually affects this chaotic deterministic system.
    """
    n_trajectories = _positive_int(n_trajectories, "n_trajectories")
    n_steps = _positive_int(n_steps, "n_steps")
    if not np.isfinite(r) or not 0 < r <= 4:
        raise ValueError("r must be finite and in (0, 4]")
    rng = np.random.default_rng(seed)
    values = np.empty((n_trajectories, n_steps + 1, 1), dtype=np.float64)
    values[:, 0, 0] = rng.beta(0.5, 0.5, size=n_trajectories)
    for t in range(n_steps):
        x = values[:, t, 0]
        values[:, t + 1, 0] = logistic_step(x, r=r)
    return values


def logistic_step(x: np.ndarray, *, r: float = 4.0) -> np.ndarray:
    """Apply ``r*x*(1-x)`` elementwise, preserving the observation shape."""
    x = np.asarray(x)
    return r * x * (1 - x)
