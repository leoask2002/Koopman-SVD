import numpy as np

from experiments.pendulum_mechanics import MechanicsReference, fit_mechanics
from experiments.pendulum_system import DT, SUBSTEPS, draw_starts, observe


def test_free_diffusion_cosine_mean_is_exact():
    # With no force and no damping, the latent dynamics are Gaussian and the
    # angle covariance has a separate closed form from sums of Gaussian kicks.
    starts = np.array([[-.4, 1.2], [2., -.8], [1., .3]])
    sigma, h = .53, 6
    m = MechanicsReference(0., 0., sigma)
    actual = m.predict(observe(starts), np.ones(h))
    dt = DT / SUBSTEPS
    expected = []
    for i in range(1, h+1):
        n = SUBSTEPS*i
        var = sigma**2 * dt**3 * n*(n+1)*(2*n+1)/6
        expected.append(np.exp(-var/2)*np.cos(starts[:, 0]+i*DT*starts[:, 1]))
    np.testing.assert_allclose(actual, np.stack(expected, axis=1), atol=1e-13)


def test_coefficients_learned_and_map_invariance():
    # Non-default unknown coefficients: learning cannot pass by substituting the
    # simulator parameters. Train on macro observations, not latent substeps.
    rng = np.random.default_rng(809)
    gs = np.array([[.1, .2, .1], [1.4, 1.1, 1.3]])
    latent = np.empty((2, 64, 4, 2))
    for k, g in enumerate(gs):
        x = draw_starts(rng, 64)
        latent[k, :, 0] = x
        q, w = x.T
        for i, gamma in enumerate(g, 1):
            for _ in range(SUBSTEPS):
                w = w + (-.72*np.sin(q)-1.36*gamma*w)*DT/SUBSTEPS
                q = q + w*DT/SUBSTEPS
            latent[k, :, i] = np.stack((q,w), axis=1)
    m = fit_mechanics(observe(latent), gs)
    np.testing.assert_allclose([m.restoring, m.damping_response], [.72,1.36], atol=1e-8)
    assert m.diffusion < 1e-8
    for sensor in ("A", "B"):
        p = m.predict(observe(latent[0, :, 0], sensor=sensor), gs[0])
        np.testing.assert_allclose(p, np.cos(latent[0, :, 1:, 0]), atol=1e-8)
