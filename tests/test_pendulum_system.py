import numpy as np

from experiments.pendulum_system import (
    SCHEDULES, conditional_endpoints, damping, draw_starts, observe,
    physics_check, reference_scores, simulate,
)


def test_independent_schedules_and_sensor_information():
    g = np.stack([damping(name) for name in SCHEDULES])
    assert np.all(g > 0) and np.unique(g, axis=0).shape == g.shape
    x = draw_starts(np.random.default_rng(11), 16)
    swapped = x.copy()
    swapped[:, 1] += 1.0
    a, b = observe(x), observe(x, sensor="B")
    assert a.shape == (16, 8)
    np.testing.assert_array_equal(a[:, :3], b[:, :3])
    assert not np.allclose(a[:, 3:], b[:, 3:])
    np.testing.assert_allclose(2*np.arctanh(a[:, 2]), x[:, 1])
    np.testing.assert_allclose(np.arctan2(a[:, 1], a[:, 0]), x[:, 0])
    np.testing.assert_array_equal(observe(x, "partial"), observe(swapped, "partial"))


def test_conditional_continuations_use_only_available_information():
    g = damping("test_smooth", 4)
    x = draw_starts(np.random.default_rng(7), 12)
    z = x.copy()
    z[:, 1] += 2
    def sample(a, view):
        return conditional_endpoints(np.random.default_rng(246), a, g, view=view, branches=4)
    p = sample(x, "partial")
    assert p.shape == (12,4,4)
    np.testing.assert_array_equal(p, sample(z, "partial"))
    assert not np.array_equal(p[:, 0], p[:, 1])
    assert not np.allclose(sample(x, "full"), sample(z, "full"))
    single = conditional_endpoints(np.random.default_rng(5), x, g, branches=1)
    sim = simulate(np.random.default_rng(5), x, g)
    np.testing.assert_allclose(single[:, 0], np.cos(sim[:, 1:, 0]))


def test_mc_mse_equals_distinct_replica_u_statistic():
    ref = np.random.default_rng(8).uniform(-1, 1, size=(9, 5, 3))
    pred = np.random.default_rng(9).uniform(-1, 1, size=(2, 9, 3))
    scores = reference_scores(pred, ref)
    e = pred[:, :, None, :] - ref
    paired = np.stack([e[:,:,i,:]*e[:,:,j,:] for i in range(5) for j in range(i+1,5)])
    np.testing.assert_allclose(scores["mse"], paired.mean(axis=(0,2)))
    np.testing.assert_allclose(scores["raw_mse"]-scores["mse"],
                               np.tile(scores["reference_noise_floor"], (2,1)))


def test_step_halving_moves_toward_independent_reference():
    d = physics_check(n=64)
    noise = d["sde_coupled_endpoint_cosine_rmse"]
    ode = d["ode_endpoint_cosine_rmse_vs_RK4_128"]
    assert noise["16_vs_32"] < .75*noise["8_vs_16"]
    assert ode["32"] < ode["16"] < ode["8"]
