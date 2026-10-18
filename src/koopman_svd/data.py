"""Explicit, boundary-safe adapters for Koopman source/target observations.

The public pairing functions return ``(x, y)`` with shape ``(n_pairs, features)``.
NumPy inputs produce NumPy arrays; PyTorch inputs preserve tensor dtype/device
and autograd. A lookback window is flattened in time-major order. No dependency
on Kooplearn is required: its current trajectory ``to_numpy()`` protocol and
legacy context dataset ``data`` protocol are supported by duck typing.
"""

from __future__ import annotations

import operator
from typing import NamedTuple, Sequence

import numpy as np
import torch


Array = np.ndarray | torch.Tensor


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _as_array(value: object) -> Array:
    if isinstance(value, (np.ndarray, torch.Tensor)):
        result = value
    elif callable(getattr(value, "to_numpy", None)):
        result = np.asarray(value.to_numpy())
    else:
        result = np.asarray(value)
    if isinstance(result, np.ndarray) and not np.issubdtype(result.dtype, np.number):
        raise TypeError("observations must have a numeric dtype")
    if result.ndim == 0:
        raise ValueError("observations must have a time or sample axis")
    return result


def context_pairs(contexts: object, *, lag: int = 1, lookback: int = 1) -> tuple[Array, Array]:
    """Extract aligned windows from explicit ``(sample, context, *state)`` data.

    ``x[i] = contexts[i, :lookback].reshape(-1)`` and
    ``y[i] = contexts[i, lag:lag+lookback].reshape(-1)``. Thus ``lag`` is the
    temporal displacement between windows, not the window length. Extra context
    entries are intentionally unused. Context rows are never joined together.
    Legacy Kooplearn ``TensorContextDataset.data`` is accepted directly. Raw
    NumPy and torch arrays are also accepted; their unrelated ``.data`` members
    are never inspected. Returned arrays/tensors preserve the input backend.
    """
    lag = _positive_int(lag, "lag")
    lookback = _positive_int(lookback, "lookback")
    if not isinstance(contexts, (np.ndarray, torch.Tensor)) and hasattr(contexts, "data"):
        contexts = contexts.data
    values = _as_array(contexts)
    if values.ndim < 2:
        raise ValueError("context data must have shape (sample, context, *state)")
    if values.shape[0] == 0:
        raise ValueError("context data must contain at least one sample")
    if values.shape[1] < lookback + lag:
        raise ValueError("context length must be at least lookback + lag")
    n_samples = values.shape[0]
    x = values[:, :lookback].reshape(n_samples, -1)
    y = values[:, lag : lag + lookback].reshape(n_samples, -1)
    if x.shape[1] == 0:
        raise ValueError("observations must contain at least one state feature")
    return x, y


def trajectory_pairs(trajectory: object, *, lag: int = 1, lookback: int = 1) -> tuple[Array, Array]:
    """Window one trajectory of shape ``(time, *state)`` into aligned pairs.

    Scalar trajectories with shape ``(time,)`` are supported. The number of
    pairs is ``time - lookback - lag + 1``. Current Kooplearn trajectories with
    ``to_numpy()`` yield NumPy arrays. Legacy context objects with ``.data`` are
    dispatched to :func:`context_pairs`; a raw array is always interpreted as
    one trajectory. For already-windowed raw arrays use ``context_pairs``
    explicitly, and for independent trajectories use ``trajectories_to_pairs``.
    """
    lag = _positive_int(lag, "lag")
    lookback = _positive_int(lookback, "lookback")
    if (
        not isinstance(trajectory, (np.ndarray, torch.Tensor))
        and not callable(getattr(trajectory, "to_numpy", None))
        and hasattr(trajectory, "data")
    ):
        return context_pairs(trajectory, lag=lag, lookback=lookback)
    values = _as_array(trajectory)
    n_pairs = values.shape[0] - lookback - lag + 1
    if n_pairs < 1:
        raise ValueError("trajectory length must be at least lookback + lag")
    if isinstance(values, torch.Tensor):
        starts = torch.arange(n_pairs, device=values.device)[:, None]
        offsets = torch.arange(lookback, device=values.device)[None, :]
    else:
        starts = np.arange(n_pairs)[:, None]
        offsets = np.arange(lookback)[None, :]
    x = values[starts + offsets].reshape(n_pairs, -1)
    y = values[starts + offsets + lag].reshape(n_pairs, -1)
    if x.shape[1] == 0:
        raise ValueError("observations must contain at least one state feature")
    return x, y


def trajectories_to_pairs(
    trajectories: Sequence[object], *, lag: int = 1, lookback: int = 1
) -> tuple[Array, Array]:
    """Concatenate paired windows without crossing trajectory boundaries.

    Pass a sequence of independent trajectories, or an array/tensor whose first
    axis indexes independent trajectories. All trajectories must have the same
    backend and flattened state dimension. Tensor dtype/device must also match.
    Split trajectories into train/validation/test groups *before* calling this
    function; splitting overlapping windows can leak observations across sets.
    """
    if len(trajectories) == 0:
        raise ValueError("at least one trajectory is required")
    pairs = [trajectory_pairs(t, lag=lag, lookback=lookback) for t in trajectories]
    tensor_backend = isinstance(pairs[0][0], torch.Tensor)
    width = pairs[0][0].shape[1]
    for x, y in pairs:
        if isinstance(x, torch.Tensor) != tensor_backend:
            raise TypeError("all trajectories must use the same NumPy or torch backend")
        if x.shape[1] != width or y.shape[1] != width:
            raise ValueError("all trajectories must have the same flattened state dimension")
        if tensor_backend and (x.dtype != pairs[0][0].dtype or x.device != pairs[0][0].device):
            raise ValueError("all torch trajectories must share dtype and device")
    if tensor_backend:
        return torch.cat([p[0] for p in pairs], dim=0), torch.cat([p[1] for p in pairs], dim=0)
    return np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs])


class TrajectorySplit(NamedTuple):
    """Disjoint groups containing original, unwindowed trajectory objects."""

    train: tuple[object, ...]
    validation: tuple[object, ...]
    test: tuple[object, ...]


def split_trajectories(
    trajectories: Sequence[object],
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> TrajectorySplit:
    """Randomly partition independent trajectories before constructing windows.

    Returns a ``TrajectorySplit(train, validation, test)`` of tuples holding
    references to the original trajectories. Each split must contain at least
    one trajectory. Fractions are rounded down for train/validation; test gets
    the remainder. This does not make correlated trajectories independent: that
    is a property of the supplied data and sampling design.
    """
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1):
        raise ValueError("train_fraction and validation_fraction must be in (0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1")
    n = len(trajectories)
    n_train = int(np.floor(n * train_fraction))
    n_validation = int(np.floor(n * validation_fraction))
    if min(n_train, n_validation, n - n_train - n_validation) < 1:
        raise ValueError("each split must contain at least one trajectory; provide more trajectories")
    order = np.random.default_rng(seed).permutation(n)
    cuts = (order[:n_train], order[n_train : n_train + n_validation], order[n_train + n_validation :])
    return TrajectorySplit(*(tuple(trajectories[int(i)] for i in ids) for ids in cuts))
