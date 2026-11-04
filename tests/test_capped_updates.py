import numpy as np
import pytest

from koopman_svd.certification import cap_forecast, certify_capped_updates, certify_improvement


@pytest.mark.parametrize("method", ["empirical_bernstein", "hoeffding"])
def test_same_forecast_sharper_certificate_and_full_range_limit(method):
    rng = np.random.default_rng(634)
    y = rng.uniform(-1, 1, (3, 256))
    base = rng.normal(size=y.shape)
    q = rng.normal(size=(2, *y.shape))
    caps = [.2, 2.]
    corrected = np.stack([cap_forecast(base, row, cap=c, bounds=(-1, 1)) for row, c in zip(q, caps)])
    generic = certify_improvement(np.concatenate((base[None], corrected)), y,
                                  baseline=0, bounds=(-1, 1), method=method)
    limited = certify_capped_updates(q, y, baseline_forecast=base, caps=caps,
                                      bounds=(-1, 1), method=method)
    np.testing.assert_allclose(limited.estimate, generic.estimate)
    assert np.all(limited.lower[1] > generic.lower[1])
    np.testing.assert_allclose(limited.lower[2], generic.lower[2])


def test_extrema_and_invalid_cap():
    base = np.array([-1., -.7])
    proposal = np.array([2., -3.])
    p = cap_forecast(base, proposal, cap=.3, bounds=(-1, 1))
    np.testing.assert_allclose(p, [-.7, -1.])
    gain = (base-1)**2 - (p-1)**2
    np.testing.assert_allclose(gain, [2*2*.3-.3**2, -(2*2*.3-.3**2)])
    with pytest.raises(ValueError):
        cap_forecast(base, proposal, cap=3, bounds=(-1, 1))


def test_threshold_bet_against_exact_binomial_null():
    from math import comb
    from koopman_svd.certification import certify_rmse_threshold
    # Prediction zero; one future observation is 1 with probability .16.
    # Single-successor score has mean tau; the e-value is a mixture of
    # true-null likelihood ratios, with exact mean one over all count outcomes.
    prob, n, delta = .16, 28, .1
    e_mean = false_reject = 0.
    for k in range(n+1):
        y = np.array([1.]*k + [0.]*(n-k))[None]
        cert = certify_rmse_threshold(np.zeros((1,1,n)), y, bounds=(0,1),
                                      tolerance=np.sqrt(prob), delta=delta)
        mass = comb(n,k)*prob**k*(1-prob)**(n-k)
        e_mean += mass*np.exp(cert.log_e_value.item())
        false_reject += mass*int(cert.accepted.item())
    np.testing.assert_allclose(e_mean, 1.)
    assert false_reject <= delta


def test_threshold_can_certify_accurate_low_variance_forecast():
    from koopman_svd.certification import certify_rmse_threshold
    p = np.zeros((2, 6, 4096))
    p[1] = .5
    y = np.zeros((6, 4096))
    accepted = certify_rmse_threshold(p, y, y, bounds=(-1,1), tolerance=.15).accepted
    assert accepted[0].all() and not accepted[1].any()
