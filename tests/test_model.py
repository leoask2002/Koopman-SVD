"""End-to-end checks against an exact, nonreversible finite Markov chain.

The transition counts are exact, so these tests isolate operator orientation,
marginal weighting, and composition from sampling and optimization error.
"""

import numpy as np
import pytest
import torch

from koopman_svd import NeuralKoopmanSVD


@pytest.fixture
def exact_chain():
    counts = np.array([[7, 2, 1], [1, 6, 3], [4, 1, 5]])
    states = np.eye(3)
    source, target = [], []
    for i in range(3):
        for j in range(3):
            source.extend([states[i]] * counts[i, j])
            target.extend([states[j]] * counts[i, j])
    x, y = np.array(source), np.array(target)
    transition = counts / counts.sum(axis=1, keepdims=True)
    return x, y, states, transition


def calibrated_chain(exact_chain):
    x, y, _, _ = exact_chain
    # A random affine neural dictionary plus a constant spans all three states.
    # Zero steps deliberately tests calibration, independently of training.
    model = NeuralKoopmanSVD(3, rank=3, hidden_dims=(), seed=19)
    return model.fit(x, y, steps=0)


def test_weighted_spectrum_and_singular_function_orientation(exact_chain):
    x, y, states, transition = exact_chain
    model = calibrated_chain(exact_chain)
    rho0, rho1 = x.mean(axis=0), y.mean(axis=0)
    reference = np.sqrt(rho0)[:, None] * transition / np.sqrt(rho1)[None, :]
    expected = np.linalg.svd(reference, compute_uv=False)
    phi, psi = model.left_functions(states), model.right_functions(states)
    np.testing.assert_allclose(model.singular_values_, expected, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(phi.T @ (rho0[:, None] * phi), np.eye(3), atol=2e-5)
    np.testing.assert_allclose(psi.T @ (rho1[:, None] * psi), np.eye(3), atol=2e-5)
    np.testing.assert_allclose(transition @ psi, phi * model.singular_values_, atol=2e-5)
    # The adjoint uses the two different empirical marginals.
    adjoint = (transition.T * rho0[None, :]) / rho1[:, None]
    np.testing.assert_allclose(adjoint @ phi, psi * model.singular_values_, atol=2e-5)


def test_nonreversible_multistep_matches_transition_powers(exact_chain):
    _, _, states, transition = exact_chain
    model = calibrated_chain(exact_chain)
    assert not np.allclose(transition, transition.T)
    for steps in (0, 1, 2, 5):
        expected = states @ np.linalg.matrix_power(transition, steps)
        np.testing.assert_allclose(model.predict(states, steps=steps), expected, atol=3e-5)
    trajectory = model.rollout(states, horizon=5)
    assert trajectory.shape == (3, 6, 3)
    np.testing.assert_allclose(trajectory[:, 0], states)
    np.testing.assert_allclose(trajectory[:, 5], np.linalg.matrix_power(transition, 5), atol=3e-5)


def test_training_actually_updates_neural_functions(exact_chain):
    x, y, _, _ = exact_chain
    model = NeuralKoopmanSVD(3, rank=3, hidden_dims=(8,), seed=7)
    model.fit(x, y, steps=0)
    before_left = model.raw_functions(x, side="left").copy()
    before_right = model.raw_functions(y, side="right").copy()
    # Refitting on identical data keeps normalization fixed, so an output
    # change here requires changes in trainable neural parameters.
    model.fit(x, y, steps=8, batch_size=16, validation_every=4, max_seconds=10)
    after_left = model.raw_functions(x, side="left")
    after_right = model.raw_functions(y, side="right")
    assert np.max(np.abs(after_left - before_left)) > 1e-7
    assert np.max(np.abs(after_right - before_right)) > 1e-7
    assert np.isfinite(model.predict(x)).all()


def test_validation_data_does_not_calibrate_model(exact_chain):
    x, y, states, _ = exact_chain
    reference = calibrated_chain(exact_chain)
    model = NeuralKoopmanSVD(3, rank=3, hidden_dims=(), seed=19)
    model.fit(x, y, steps=0, validation_data=(x + 100, y - 100))
    np.testing.assert_allclose(model.predict(states), reference.predict(states), atol=1e-12)
    np.testing.assert_allclose(model.singular_values_, reference.singular_values_, atol=1e-12)


def test_constant_mode_preserves_target_mean_without_drift():
    x = np.linspace(-1, 1, 31)[:, None]
    y = 2 * x + 5
    model = NeuralKoopmanSVD(1, rank=1, seed=3).fit(x, y, steps=0)
    np.testing.assert_allclose(model.singular_values_, [1.0], atol=1e-12)
    for steps in (1, 10, 100):
        np.testing.assert_allclose(model.predict(x, steps=steps), np.full_like(y, y.mean()), atol=1e-11)


def test_serialization_preserves_operator_and_tensor_predictions(exact_chain, tmp_path):
    _, _, states, _ = exact_chain
    model = calibrated_chain(exact_chain)
    checkpoint = tmp_path / "model.pt"
    model.save(checkpoint)
    restored = NeuralKoopmanSVD.load(checkpoint)
    np.testing.assert_allclose(restored.predict(states, steps=4), model.predict(states, steps=4), atol=1e-12)
    np.testing.assert_allclose(restored.singular_values_, model.singular_values_, atol=1e-12)
    tensor_prediction = restored.predict(torch.as_tensor(states, dtype=torch.float64), steps=2)
    assert isinstance(tensor_prediction, torch.Tensor)
    np.testing.assert_allclose(tensor_prediction.detach().cpu().numpy(), restored.predict(states, steps=2))


def test_invalid_or_nonfinite_input_is_rejected(exact_chain):
    x, y, _, _ = exact_chain
    model = NeuralKoopmanSVD(3, rank=3, hidden_dims=(), seed=19)
    invalid = x.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError):
        model.fit(invalid, y, steps=0)
    model.fit(x, y, steps=0)
    with pytest.raises(ValueError):
        model.predict(x, steps=-1)
    with pytest.raises(ValueError):
        model.predict(np.ones((4, 2)))
