from types import SimpleNamespace

import numpy as np
import pytest
import torch

from koopman_svd.benchmarks import (
    gaussian_ar1_trajectories,
    logistic_step,
    logistic_trajectories,
    triangular_features,
    triangular_step,
    triangular_trajectories,
)
from koopman_svd.data import context_pairs, split_trajectories, trajectories_to_pairs, trajectory_pairs


@pytest.mark.parametrize("tensor", [False, True])
def test_lagged_lookback_is_temporally_aligned_and_flattened(tensor):
    values = np.arange(20, dtype=np.float64).reshape(10, 2)
    data = torch.tensor(values, requires_grad=True) if tensor else values
    x, y = trajectory_pairs(data, lag=2, lookback=3)
    assert x.shape == y.shape == (6, 6)
    if tensor:
        assert x.dtype == data.dtype and x.device == data.device
        torch.testing.assert_close(x[0], data[:3].reshape(-1))
        torch.testing.assert_close(y[-1], data[7:10].reshape(-1))
        (x.sum() + y.sum()).backward()
        assert data.grad is not None and torch.all(data.grad > 0)
    else:
        np.testing.assert_array_equal(x[0], values[:3].reshape(-1))
        np.testing.assert_array_equal(y[-1], values[7:10].reshape(-1))


def test_context_rows_are_windows_not_a_trajectory_of_states():
    raw = np.arange(3 * 5 * 2).reshape(3, 5, 2)
    legacy_dataset = SimpleNamespace(data=raw)
    x, y = trajectory_pairs(legacy_dataset, lag=2, lookback=2)
    assert x.shape == y.shape == (3, 4)
    np.testing.assert_array_equal(x, raw[:, :2].reshape(3, 4))
    np.testing.assert_array_equal(y, raw[:, 2:4].reshape(3, 4))
    expected = context_pairs(raw, lag=2, lookback=2)
    for actual, reference in zip((x, y), expected):
        np.testing.assert_array_equal(actual, reference)


def test_torch_context_does_not_use_tensor_data_or_detach_gradients():
    raw = torch.randn(4, 3, 2, dtype=torch.float64, requires_grad=True)
    x, y = context_pairs(raw, lookback=2)
    (x.square().sum() + y.square().sum()).backward()
    assert raw.grad is not None and raw.grad.abs().sum() > 0


def test_modern_trajectory_numpy_protocol():
    class Trajectory:
        def to_numpy(self):
            return np.arange(12).reshape(6, 2)

    x, y = trajectory_pairs(Trajectory())
    np.testing.assert_array_equal(y, x + 2)
    assert x.shape == (5, 2)


def test_installed_kooplearn_current_trajectory_adapter():
    datasets = pytest.importorskip("kooplearn.datasets")
    if not hasattr(datasets, "make_linear_system"):
        pytest.skip("this integration test targets the optional Kooplearn 2.x API")
    a = np.array([[0.7, -0.2], [0.1, 0.4]])
    trajectory = datasets.make_linear_system([1.0, -0.5], a, n_steps=8, noise=0.0)
    x, y = trajectory_pairs(trajectory)
    assert isinstance(x, np.ndarray)
    assert x.shape == y.shape == (8, 2)
    np.testing.assert_allclose(y, x @ a.T, atol=1e-14)


def test_split_before_windowing_keeps_trajectory_ids_disjoint():
    trajectories = [np.column_stack((np.full(6, i), np.arange(6))) for i in range(12)]
    partitions = split_trajectories(trajectories, train_fraction=0.5, validation_fraction=0.25, seed=7)
    assert list(map(len, partitions)) == [6, 3, 3]
    ids_by_partition = []
    for group in partitions:
        x, y = trajectories_to_pairs(group, lag=2, lookback=2)
        assert len(x) == len(group) * 3
        np.testing.assert_array_equal(x[:, 0], y[:, 0])
        np.testing.assert_array_equal(x[:, 0], x[:, 2])
        ids_by_partition.append(set(x[:, 0]))
    assert set.union(*ids_by_partition) == set(range(12))
    assert not (ids_by_partition[0] & ids_by_partition[1])
    assert not (ids_by_partition[0] & ids_by_partition[2])
    assert not (ids_by_partition[1] & ids_by_partition[2])
    again = split_trajectories(trajectories, train_fraction=0.5, validation_fraction=0.25, seed=7)
    assert [id(t) for t in partitions.train] == [id(t) for t in again.train]


def test_independent_trajectories_do_not_create_boundary_pairs():
    sequences = np.array([[0, 1, 2], [100, 101, 102]])
    x, y = trajectories_to_pairs(sequences)
    np.testing.assert_array_equal(x[:, 0], [0, 1, 100, 101])
    np.testing.assert_array_equal(y[:, 0], [1, 2, 101, 102])
    xt, yt = trajectories_to_pairs(torch.as_tensor(sequences))
    torch.testing.assert_close(xt, torch.as_tensor(x))
    torch.testing.assert_close(yt, torch.as_tensor(y))


@pytest.mark.parametrize("kwargs", [{"lag": 0}, {"lookback": -1}, {"lag": 1.5}, {"lookback": True}])
def test_invalid_window_specification(kwargs):
    with pytest.raises(ValueError):
        trajectory_pairs(np.arange(5), **kwargs)


def test_invalid_data_shapes_and_split_sizes():
    with pytest.raises(ValueError, match="length"):
        trajectory_pairs(np.zeros((2, 3)), lookback=2)
    with pytest.raises(ValueError, match="context length"):
        context_pairs(np.zeros((5, 1, 2)))
    with pytest.raises(TypeError, match="same"):
        trajectories_to_pairs([np.ones((4, 2)), torch.ones((4, 2))])
    with pytest.raises(ValueError, match="each split"):
        split_trajectories([np.arange(3), np.arange(3)])
    with pytest.raises(ValueError, match="less than 1"):
        split_trajectories([np.arange(3)] * 10, train_fraction=0.8, validation_fraction=0.3)


def test_gaussian_reference_covariance_and_nonsymmetric_orientation():
    trajectories, a = gaussian_ar1_trajectories(n_trajectories=50000, n_steps=1, seed=81)
    x, y = trajectories[:, 0], trajectories[:, 1]
    assert not np.allclose(a, a.T)
    np.testing.assert_allclose(np.linalg.svd(a, compute_uv=False), [0.85, 0.55, 0.25], atol=1e-14)
    np.testing.assert_allclose(x.T @ x / len(x), np.eye(3), atol=0.02)
    np.testing.assert_allclose(y.T @ y / len(y), np.eye(3), atol=0.02)
    np.testing.assert_allclose(x.T @ y / len(x), a.T, atol=0.02)
    # A transposition bug is materially larger than Monte Carlo tolerance.
    assert np.linalg.norm(x.T @ y / len(x) - a) > 0.3


def test_triangular_exact_dictionary_closure_and_reproducibility():
    a, b = 0.73, 0.41
    trajectories = triangular_trajectories(n_trajectories=8, n_steps=4, seed=21, a=a, b=b)
    np.testing.assert_array_equal(trajectories, triangular_trajectories(8, 4, seed=21, a=a, b=b))
    x, y = trajectories_to_pairs(trajectories)
    operator = np.array([[a, 0, 0], [0, b, 0], [0, a * a - b, a * a]])
    np.testing.assert_allclose(triangular_features(x) @ operator, triangular_features(y), atol=1e-15)
    np.testing.assert_allclose(triangular_step(x, a=a, b=b), y, atol=1e-15)


def test_logistic_dynamics_and_parameter_validation():
    trajectories = logistic_trajectories(n_trajectories=12, n_steps=5, r=3.7, seed=42)
    x, y = trajectories_to_pairs(trajectories)
    np.testing.assert_allclose(y, logistic_step(x, r=3.7), atol=1e-15)
    assert trajectories.shape == (12, 6, 1)
    assert np.all((trajectories >= 0) & (trajectories <= 1))
    with pytest.raises(ValueError):
        logistic_trajectories(r=4.1)
    with pytest.raises(ValueError):
        gaussian_ar1_trajectories(matrix=np.eye(2))
