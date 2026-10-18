"""Differentiable joint-nesting NeuralSVD objective.

Training optimizes moments; this module never factors a matrix.  Independent
source and target marginal batches give an unbiased gradient of the empirical
product-of-marginals objective when their indices are drawn independently with
replacement.  The paired batch estimates the cross moment separately.
"""

from __future__ import annotations

import torch
from torch import Tensor


def nesting_masks(rank: int, *, dtype=None, device=None, weights=None):
    """Return tail weights ``a`` and ``W[i,j] = a[max(i,j)]``.

    ``weights[l]`` weights the objective for the first ``l+1`` functions.
    Uniform prefix weights, summing to one, are the default.  Positive weights
    distinguish every nested prefix; zero weights are accepted for ablations.
    """
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("rank must be a positive integer")
    dtype = dtype or torch.get_default_dtype()
    if weights is None:
        prefixes = torch.full((rank,), 1.0 / rank, dtype=dtype, device=device)
    else:
        prefixes = torch.as_tensor(weights, dtype=dtype, device=device)
        if prefixes.shape != (rank,):
            raise ValueError("weights must contain one coefficient per prefix")
        if not bool(torch.isfinite(prefixes).all()) or bool((prefixes < 0).any()):
            raise ValueError("weights must be finite and nonnegative")
        if not bool((prefixes > 0).any()):
            raise ValueError("at least one prefix weight must be positive")
    a = prefixes.flip(0).cumsum(0).flip(0)
    indices = torch.arange(rank, device=device)
    return a, a[torch.maximum(indices[:, None], indices[None, :])]


def neural_svd_loss(
    f_pair: Tensor,
    g_pair: Tensor,
    f_marginal: Tensor,
    g_marginal: Tensor,
    *,
    weights=None,
) -> Tensor:
    """Weighted sum of all prefix low-rank approximation objectives.

    ``f_pair`` is evaluated at source states X, ``g_pair`` at their paired
    target states Y.  ``f_marginal`` and ``g_marginal`` must come from independent
    index draws for unbiased stochastic optimization of the empirical loss.
    Its constant Hilbert--Schmidt term is omitted.  The objective is meaningful
    as a finite-rank moment objective even when an infinite-dimensional
    deterministic Koopman operator is not Hilbert--Schmidt.
    """
    tensors = (f_pair, g_pair, f_marginal, g_marginal)
    if any(not isinstance(t, Tensor) or t.ndim != 2 or t.shape[0] < 1 for t in tensors):
        raise ValueError("all feature batches must be nonempty 2D tensors")
    rank = f_pair.shape[1]
    if rank < 1 or any(t.shape[1] != rank for t in tensors):
        raise ValueError("feature batches must have the same positive rank")
    if f_pair.shape[0] != g_pair.shape[0]:
        raise ValueError("paired source and target batches must have equal length")
    if any(t.dtype != f_pair.dtype or t.device != f_pair.device for t in tensors):
        raise ValueError("feature batches must share dtype and device")
    a, w = nesting_masks(rank, dtype=f_pair.dtype, device=f_pair.device, weights=weights)
    c_f = f_marginal.T @ f_marginal / f_marginal.shape[0]
    c_g = g_marginal.T @ g_marginal / g_marginal.shape[0]
    paired = (f_pair * g_pair).mean(dim=0)
    return -2 * torch.sum(a * paired) + torch.sum(w * c_f * c_g)
