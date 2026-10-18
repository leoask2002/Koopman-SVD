"""Training isolation and honest handling of redundant learned dictionaries."""

import numpy as np
import pytest
import torch

from koopman_svd import NeuralKoopmanSVD


@pytest.mark.parametrize("tensor", [False, True])
def test_complex_states_are_rejected_without_discarding_information(tensor):
    x = np.array([[1 + 2j], [3 - 1j]])
    if tensor:
        x = torch.as_tensor(x)
    model = NeuralKoopmanSVD(1, rank=1)
    with pytest.raises(ValueError, match="real-valued"):
        model.fit(x, x, steps=0)


def test_redundant_affine_dictionary_truncates_and_roundtrips(tmp_path):
    x = np.linspace(-2, 2, 50)[:, None]
    y = 0.65 * x + 0.25
    # Five affine functions of one scalar have at most two independent modes.
    model = NeuralKoopmanSVD(1, rank=5, hidden_dims=(), seed=9).fit(x, y, steps=0)
    assert model.effective_rank_ == 2
    assert model.calibration_summary_["rank_truncated"] is True
    assert model.calibration_summary_["requested_rank"] == 5
    assert model.calibration_summary_["retained_rank"] == 2
    assert model.raw_functions(x).shape == (len(x), 5)
    assert model.left_functions(x).shape == (len(x), 2)
    np.testing.assert_allclose(model.left_functions(x).T @ model.left_functions(x) / len(x), np.eye(2), atol=1e-11)
    expected = 0.65**4 * x + 0.25 * sum(0.65**j for j in range(4))
    np.testing.assert_allclose(model.predict(x, steps=4), expected, atol=1e-11)
    model.save(tmp_path / "redundant.pt")
    restored = NeuralKoopmanSVD.load(tmp_path / "redundant.pt")
    assert restored.effective_rank_ == 2
    np.testing.assert_allclose(restored.predict(x, steps=4), expected, atol=1e-11)
    assert restored.diagnostics(x, y)["calibration"]["rank_truncated"] is True


def test_local_randomness_preserves_global_generators_and_repeats_results():
    x = np.linspace(-1, 1, 40)[:, None]
    y = 0.75 * x + 0.1 * x**2
    torch_state = torch.random.get_rng_state().clone()
    numpy_state = np.random.get_state()
    first = NeuralKoopmanSVD(1, rank=3, hidden_dims=(8,), seed=42)
    first.fit(x, y, steps=4, batch_size=16, validation_every=2, max_seconds=30)
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    after_numpy = np.random.get_state()
    assert numpy_state[0] == after_numpy[0]
    np.testing.assert_array_equal(numpy_state[1], after_numpy[1])
    assert numpy_state[2:] == after_numpy[2:]
    second = NeuralKoopmanSVD(1, rank=3, hidden_dims=(8,), seed=42)
    second.fit(x, y, steps=4, batch_size=16, validation_every=2, max_seconds=30)
    np.testing.assert_array_equal(first.raw_functions(x), second.raw_functions(x))
    np.testing.assert_array_equal(first.singular_values_, second.singular_values_)
    np.testing.assert_array_equal(first.predict(x), second.predict(x))
