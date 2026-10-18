import pytest
import torch

from koopman_svd.losses import nesting_masks, neural_svd_loss


def test_joint_nesting_matches_explicit_prefixes_and_gradients():
    generator = torch.Generator().manual_seed(29)
    f, g, fm, gm = [
        torch.randn(n, 4, generator=generator, dtype=torch.float64, requires_grad=True)
        for n in (7, 7, 5, 9)
    ]
    weights = torch.tensor([0.15, 0.05, 0.3, 0.5], dtype=torch.float64)
    actual = neural_svd_loss(f, g, fm, gm, weights=weights)
    expected = torch.zeros((), dtype=torch.float64)
    for size, weight in enumerate(weights, start=1):
        paired = (f[:, :size] * g[:, :size]).sum(dim=1).mean()
        # Explicit Cartesian product of independent marginal observations.
        metric = (fm[:, :size] @ gm[:, :size].T).square().mean()
        expected = expected + weight * (-2 * paired + metric)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    actual_gradients = torch.autograd.grad(actual, (f, g, fm, gm), retain_graph=True)
    expected_gradients = torch.autograd.grad(expected, (f, g, fm, gm))
    for actual_grad, expected_grad in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-13, atol=1e-13)


def test_independent_marginal_sampling_has_unbiased_metric_and_gradient():
    # Finite population lets us enumerate *all* draws, with no Monte Carlo error.
    f = torch.tensor([[1.0], [3.0]], dtype=torch.float64, requires_grad=True)
    g = torch.tensor([[1.0], [3.0]], dtype=torch.float64, requires_grad=True)
    z = torch.zeros((1, 1), dtype=torch.float64)
    exact = neural_svd_loss(z, z, f, g)
    independent = torch.stack([
        neural_svd_loss(z, z, f[i : i + 1], g[j : j + 1])
        for i in range(2) for j in range(2)
    ]).mean()
    correlated = torch.stack([
        neural_svd_loss(z, z, f[i : i + 1], g[i : i + 1]) for i in range(2)
    ]).mean()
    torch.testing.assert_close(independent, exact)
    assert exact.item() == 25.0
    assert correlated.item() == 41.0
    independent_grad = torch.autograd.grad(independent, (f, g), retain_graph=True)
    exact_grad = torch.autograd.grad(exact, (f, g))
    for actual, expected in zip(independent_grad, exact_grad):
        torch.testing.assert_close(actual, expected)


def test_loss_passes_numerical_autograd_check():
    generator = torch.Generator().manual_seed(11)
    arguments = tuple(
        torch.randn(n, 3, generator=generator, dtype=torch.float64, requires_grad=True)
        for n in (3, 3, 4, 2)
    )
    assert torch.autograd.gradcheck(neural_svd_loss, arguments, fast_mode=True)


def test_mask_prefix_weight_semantics():
    a, mask = nesting_masks(3, weights=[1.0, 2.0, 4.0], dtype=torch.float64)
    torch.testing.assert_close(a, torch.tensor([7.0, 6.0, 4.0], dtype=torch.float64))
    torch.testing.assert_close(mask, torch.tensor([[7, 6, 4], [6, 6, 4], [4, 4, 4]], dtype=torch.float64))


@pytest.mark.parametrize("weights", [[0.0, 0.0], [-1.0, 1.0], [1.0], [float("nan"), 1.0]])
def test_invalid_prefix_weights(weights):
    with pytest.raises(ValueError):
        nesting_masks(2, weights=weights)


def test_malformed_batches_are_rejected():
    f = torch.ones(3, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="equal length"):
        neural_svd_loss(f, f[:2], f, f)
    with pytest.raises(ValueError, match="positive rank"):
        neural_svd_loss(f, f, f[:, :1], f)
    with pytest.raises(ValueError, match="dtype"):
        neural_svd_loss(f, f, f.float(), f)
    with pytest.raises(ValueError, match="nonempty"):
        neural_svd_loss(f[:0], f[:0], f, f)
