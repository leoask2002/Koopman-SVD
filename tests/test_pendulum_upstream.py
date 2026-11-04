"""Opt-in parity/integration checks on the exact external MIT checkout."""
import os

import numpy as np
import pytest
import torch

from koopman_svd.losses import neural_svd_loss
from experiments.pendulum_upstream import _upstream, fit_official


@pytest.fixture
def checkout():
    path = os.environ.get("KOOPMAN_UPSTREAM")
    if not path:
        pytest.skip("optional pinned upstream checkout not configured")
    return path


def test_authors_joint_objective_and_gradient_on_identical_grams(checkout):
    """Agreement is conditional on identical Gram batches; actual sampling differs."""
    _, loss_cls, _, _, _ = _upstream(checkout)
    rng = torch.Generator().manual_seed(112)
    fs = [torch.randn((12, 4), generator=rng, dtype=torch.double, requires_grad=True)
          for _ in range(2)]
    local = neural_svd_loss(*fs, *fs)
    official = loss_cls(4, nesting="jnt", reg_weight=0).double()(*fs)
    torch.testing.assert_close(official, local)
    gl = torch.autograd.grad(local, fs)
    go = torch.autograd.grad(official, fs)
    for left, right in zip(gl, go):
        torch.testing.assert_close(left, right)


def test_native_homogeneous_predictions_do_not_depend_on_future_schedule(checkout):
    rng = np.random.default_rng(998)
    train = rng.normal(size=(3, 60, 4, 3))
    val = rng.normal(size=(2, 30, 4, 3))
    td = rng.uniform(size=(3,3))
    vd = rng.uniform(size=(2,3))
    candidates = fit_official(train, td, val, vd, seed=49, matched_rank=4, steps=2)
    assert set(candidates) == {"upstream_cca", "upstream_edmd", "upstream_prefix"}
    initial = val[0, :, 0]
    for key, model in candidates.items():
        p = model.predict(initial, vd[0])
        assert p.shape == (30, 3) and np.max(np.abs(p)) <= 1.
        if key != "upstream_prefix":
            np.testing.assert_equal(p, model.predict(initial, vd[1]))
