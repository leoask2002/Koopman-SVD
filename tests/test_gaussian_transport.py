import numpy as np

from experiments.gaussian_transport import compositions, fit_models, ols, score, simulate, system


def test_reverse_composition_equals_direct_two_time_conditional():
    """Use the joint Gaussian cross-covariance, not products of reverse maps, as oracle."""
    law = system("drifting")
    mean, cov = law.marginals()
    h = 6
    (forward, _), (backward, offset) = compositions(law.maps(), h)
    for t, f in enumerate(forward):
        cross = cov[t] @ f.T
        expected = np.linalg.solve(cov[t + h], cross.T).T
        np.testing.assert_allclose(backward[t], expected, atol=1e-13)
        np.testing.assert_allclose(offset[t], mean[t] - expected @ mean[t + h], atol=1e-13)


def test_time_local_gaussian_fit_recovers_direct_reverse_regression():
    paths = simulate(system("shifted_homogeneous"), np.random.default_rng(4), 64)
    fits, _ = fit_models(paths)
    _, _, backward, intercept = fits[1.0]
    for t in range(paths.shape[1] - 1):
        a, b = ols(paths[:, t + 1], paths[:, t])
        np.testing.assert_allclose(backward[t], a, atol=1e-12)
        np.testing.assert_allclose(intercept[t], b, atol=1e-12)


def test_exact_error_integral_agrees_with_degree_two_gaussian_cubature():
    law = system("drifting")
    paths = simulate(law, np.random.default_rng(25), 32)
    fit = fit_models(paths)[0][0.5]
    mu, c = law.marginals()
    metrics = score(fit, law.maps(), mu, c)
    h = 3
    (pred, pred_b), _ = compositions(fit, h)
    (true, true_b), _ = compositions(law.maps(), h)
    direct_errors = []
    for t in range(len(pred)):
        # 2d symmetric sigma points reproduce exactly the mean and covariance.
        d = c.shape[-1]
        half = np.sqrt(d) * np.linalg.cholesky(c[t]).T
        points = mu[t] + np.concatenate([half, -half], axis=0)
        forecast = points @ pred[t].T + pred_b[t]
        actual = points @ true[t].T + true_b[t]
        direct_errors.append(np.mean((forecast - actual)**2))
    np.testing.assert_allclose(metrics["3"]["forward"], np.sqrt(np.mean(direct_errors)), atol=1e-13)


def test_smoothed_regression_endpoints_reproduce_baseline_maps():
    from experiments.gaussian_smoothing import local_smooth

    paths = simulate(system("drifting"), np.random.default_rng(12), 24)
    fits, _ = fit_models(paths)
    for w, bandwidth in ((0.0, np.inf), (1.0, 0.0)):
        got = local_smooth(paths, bandwidth)
        for field, expected in zip(got, fits[w]):
            np.testing.assert_allclose(field, expected, atol=1e-12)
