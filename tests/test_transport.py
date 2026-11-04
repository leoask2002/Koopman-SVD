"""Study properties of composition, not the internal whitening convention."""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from koopman_svd.transport import TimeIndexedKoopmanSVD


def exact_paths():
    # Enumerate independent paths with integer multiplicities, so all
    # empirical conditional laws and all three changing marginals are exact.
    initial = np.array([2, 3, 5])
    transitions = np.array([[[8, 1, 1], [2, 5, 3], [1, 2, 7]],
                            [[1, 7, 2], [4, 1, 5], [3, 4, 3]]])
    rows = []
    for i, j, k in product(range(3), repeat=3):
        rows.extend([[i, j, k]] * int(initial[i] * transitions[0, i, j] * transitions[1, j, k]))
    return np.asarray(rows, dtype=float)[:, :, None], transitions / 10


def onehot(x):
    return np.eye(3)[x[:, 0].astype(int)]


def test_nonstationary_full_rank_recovers_entire_two_step_kernel_and_requires_bridge():
    paths, (p, q) = exact_paths()
    states = np.arange(3)[:, None]
    est = TimeIndexedKoopmanSVD.fit(paths, rank=3, left_features=onehot, right_features=onehot)
    np.testing.assert_allclose(est.predict(states, observable=onehot), p, atol=1e-12)
    np.testing.assert_allclose(est.predict(states, start=1, observable=onehot), q, atol=1e-12)
    np.testing.assert_allclose(est.predict(states, horizon=2, observable=onehot), p @ q, atol=1e-12)
    np.testing.assert_allclose(est.predict(states, start=2, horizon=0), states)
    # Dropping the overlap is not a valid composition even at zero sampling
    # error. The discrepancy cannot be repaired by increasing training compute.
    unbridged = replace(est, bridges=(np.eye(3),))
    assert np.linalg.norm(unbridged.predict(states, horizon=2, observable=onehot) - p @ q) > .01


def test_independent_changes_of_basis_leave_predictions_unchanged():
    paths, _ = exact_paths()
    values = np.arange(3)[:, None]
    rng = np.random.default_rng(548)
    maps = []
    for _ in range(4):
        u, _, vh = np.linalg.svd(rng.normal(size=(3, 3)))
        mat = u @ np.diag([.2, 1., 3.]) @ vh
        offset = rng.uniform(size=3)
        maps.append(lambda x, mat=mat, offset=offset: onehot(x) @ mat + offset)
    base = TimeIndexedKoopmanSVD.fit(paths, rank=2, left_features=onehot, right_features=onehot)
    changed = TimeIndexedKoopmanSVD.fit(paths, rank=2, left_features=maps[:2], right_features=maps[2:])
    np.testing.assert_allclose(base.predict(values, horizon=2, observable=onehot),
                               changed.predict(values, horizon=2, observable=onehot), atol=1e-12)


def test_prefixes_heterogeneous_ranks_and_singular_dictionaries():
    paths, _ = exact_paths()
    redundant = lambda x: np.column_stack((np.full(len(x), .1), onehot(x), 2 * onehot(x)))
    all_modes = TimeIndexedKoopmanSVD.fit(paths, left_features=redundant, right_features=redundant)
    low = all_modes.truncate([3, 2])
    direct = TimeIndexedKoopmanSVD.fit(paths, rank=[3, 2], left_features=redundant, right_features=redundant)
    assert low.bridges[0].shape == (3, 2)
    np.testing.assert_allclose(low.predict(paths[:5, 0], horizon=2),
                               direct.predict(paths[:5, 0], horizon=2))
    constant = lambda x: np.full(len(x), .1)
    np.testing.assert_array_equal(low.predict(paths[:5, 0], horizon=2, observable=constant), [[.1]] * 5)
    flat = TimeIndexedKoopmanSVD.fit(paths, rank=4, left_features=constant)
    np.testing.assert_array_equal(flat.steps[0].singular_values, [1.])
    np.testing.assert_allclose(flat.predict(paths[:5, 0], horizon=2), paths[:, -1, :].mean(0) * np.ones((5,1)))
    # Changing the caller's training array cannot alter the fitted decoder.
    before = direct.predict(paths[:2, 0], horizon=2)
    x = paths[:2, 0].copy()
    paths[:] = 1000
    np.testing.assert_array_equal(direct.predict(x, horizon=2), before)


def test_neural_adapter_uses_distinct_frozen_dictionaries_per_step():
    paths, (p, q) = exact_paths()

    class Dictionary:
        def __init__(self, scale):
            self.scale = scale

        def raw_functions(self, x, *, side):
            return onehot(x) * (self.scale if side == "left" else 1 / self.scale)

    est = TimeIndexedKoopmanSVD.from_neural_models(paths, [Dictionary(2.), Dictionary(.3)], rank=3)
    np.testing.assert_allclose(est.predict(np.arange(3)[:, None], horizon=2, observable=onehot), p @ q, atol=1e-12)


def test_invalid_shape_or_unfitted_horizon_is_rejected():
    paths, _ = exact_paths()
    with pytest.raises(ValueError, match="one entry per transition"):
        TimeIndexedKoopmanSVD.fit(paths, rank=[1, 2, 2])
    with pytest.raises(ValueError, match="finite array"):
        TimeIndexedKoopmanSVD.fit(paths[:, :1])
    fit = TimeIndexedKoopmanSVD.fit(paths)
    with pytest.raises(ValueError, match="beyond"):
        fit.predict(paths[:3, 0], horizon=3)
    with pytest.raises(ValueError, match="correct number of rows"):
        fit.predict(paths[:3, 0], observable=lambda x: x[:2])
