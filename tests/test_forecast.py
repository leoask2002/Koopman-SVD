import numpy as np
import pytest
import torch

from koopman_svd import NeuralKoopmanSVD
from koopman_svd.forecast import StateAugmentedEDMD


@pytest.fixture
def affine_case():
    rng = np.random.default_rng(400)
    x = rng.normal(size=(160, 2))
    a = np.array([[0.7, -0.2], [0.15, 0.4]])
    offset = np.array([0.3, -0.1])
    y = x @ a.T + offset
    # A constant-only SVD deliberately cannot retain the raw state; the
    # additional forecasting head must recover the dynamics through augmentation.
    model = NeuralKoopmanSVD(2, 1, hidden_dims=()).fit(x, y, steps=0)
    return x, y, a, offset, model


def test_state_augmentation_recovers_affine_dynamics_and_powers(affine_case):
    x, y, a, offset, model = affine_case
    head = StateAugmentedEDMD(model).fit(x, y)
    assert head.effective_rank_ == 3  # redundant learned constant is dropped
    initial = x[:12] + 0.31
    truth = initial.copy()
    for horizon in range(11):
        np.testing.assert_allclose(head.predict(initial, steps=horizon), truth, atol=2e-12)
        truth = truth @ a.T + offset
    expected_eigenvalues = np.r_[1, np.linalg.eigvals(a)]
    np.testing.assert_allclose(np.sort_complex(head.spectral_diagnostics()["eigenvalues"]),
                               np.sort_complex(expected_eigenvalues), atol=1e-12)
    assert head.spectral_diagnostics()["spectral_radius"] == pytest.approx(1.0)


def test_head_does_not_modify_neural_singular_functions(affine_case):
    x, y, _, _, model = affine_case
    before = model.left_functions(x).copy()
    spectrum = model.singular_values_.copy()
    StateAugmentedEDMD(model).fit(x, y)
    np.testing.assert_array_equal(model.left_functions(x), before)
    np.testing.assert_array_equal(model.singular_values_, spectrum)


def test_prediction_does_not_refit_scaling_and_preserves_tensor_type(affine_case):
    x, y, _, _, model = affine_case
    head = StateAugmentedEDMD(model).fit(x, y)
    mean, scale = head.mean_.copy(), head.scale_.copy()
    batch = x[:8] + 100
    whole = head.predict(batch)
    separate = np.concatenate([head.predict(row[None]) for row in batch])
    np.testing.assert_allclose(whole, separate, atol=1e-12)
    np.testing.assert_array_equal(mean, head.mean_)
    np.testing.assert_array_equal(scale, head.scale_)
    tensor = torch.tensor(batch, dtype=torch.float64, requires_grad=True)
    actual = head.predict(tensor, steps=3)
    assert isinstance(actual, torch.Tensor) and actual.dtype == tensor.dtype
    assert actual.device == tensor.device and not actual.requires_grad
    np.testing.assert_allclose(actual.numpy(), head.predict(batch, steps=3))


def test_rank_and_ridge_are_explicit_and_reported(affine_case):
    x, y, _, _, model = affine_case
    head = StateAugmentedEDMD(model, rank=2, ridge=1e-3).fit(x, y)
    assert head.operator_.shape == (2, 2)
    assert head.spectral_diagnostics()["ridge"] == 1e-3
    assert np.isfinite(head.predict(x, steps=10)).all()
    with pytest.raises(ValueError, match="supported"):
        StateAugmentedEDMD(model, rank=4).fit(x, y)


def test_invalid_or_unfitted_head_is_rejected(affine_case):
    x, y, _, _, model = affine_case
    with pytest.raises(ValueError):
        StateAugmentedEDMD(model, ridge=-1)
    head = StateAugmentedEDMD(model)
    with pytest.raises(RuntimeError):
        head.predict(x)
    with pytest.raises(ValueError):
        head.fit(x, y[:-1])
    head.fit(x, y)
    for steps in (-1, True, 0.2):
        with pytest.raises(ValueError):
            head.predict(x, steps=steps)
    with pytest.raises(ValueError):
        head.predict([[np.nan, 1]])


def test_unstable_long_horizon_overflow_is_reported():
    x = np.linspace(-1, 1, 40)[:, None]
    y = 2 * x
    model = NeuralKoopmanSVD(1, 1).fit(x, y, steps=0)
    head = StateAugmentedEDMD(model).fit(x, y)
    assert head.spectral_diagnostics()["spectral_radius"] == pytest.approx(2.0)
    with pytest.raises(FloatingPointError, match="overflowed"):
        head.predict(x, steps=10000)


def test_complex_states_are_rejected_without_discarding_imaginary_part(affine_case):
    x, y, _, _, model = affine_case
    head = StateAugmentedEDMD(model).fit(x, y)
    for complex_x in (x.astype(complex) + 1j, torch.as_tensor(x.astype(complex) + 1j)):
        with pytest.raises(ValueError, match="complex-valued"):
            head.predict(complex_x)
