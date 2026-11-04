import math

import numpy as np
import pytest

from koopman_svd.certification import certify_forecasts, certify_improvement, select_horizon


def test_replicated_outcomes_remove_conditional_variance():
    # Enumerate the four equally likely pairs of independent Bernoulli(1/2) draws.
    y = np.tile([0, 0, 1, 1], 1024)[None, :]
    z = np.tile([0, 1, 0, 1], 1024)[None, :]
    p = np.stack([np.full_like(y, 0.5, dtype=float), np.zeros_like(y)])
    pair = certify_forecasts(p, y, z, bounds=(0, 1))
    single = certify_forecasts(p, y, bounds=(0, 1))
    np.testing.assert_allclose(pair.estimate[:, 0], [0.0, 0.25])
    np.testing.assert_allclose(single.estimate - pair.estimate, 0.25)
    assert pair.upper[0, 0] < 0.1 < single.upper[0, 0]
    assert pair.n_starts == y.shape[-1] and pair.paired and not single.paired


def test_only_the_clipped_forecast_is_certified_and_units_rescale():
    y = np.broadcast_to(np.arange(100) % 2, (2, 100))
    p = np.stack([y + 4, y - 4])
    clipped = np.clip(p, 0, 1)
    result = certify_forecasts(p, y, 1 - y, bounds=(0, 1))
    comp = certify_forecasts(clipped, y, 1 - y, bounds=(0, 1))
    scaled = certify_forecasts(2*p-3, 2*y-3, 2*(1-y)-3, bounds=(-3, -1))
    np.testing.assert_array_equal(result.upper, comp.upper)
    np.testing.assert_allclose(scaled.upper, 4*comp.upper)
    np.testing.assert_allclose(scaled.estimate, 4*comp.estimate)


def test_negative_cross_products_are_not_discarded_before_averaging():
    y = np.zeros((1, 512))
    p = np.full((1, 1, 512), 0.5)
    cert = certify_forecasts(p, y, 1 - y, bounds=(0, 1), method="hoeffding")
    assert cert.estimate.item() == -0.25
    assert cert.upper.item() == 0.0


def test_horizon_abstention_prefix_and_rank_tiebreak():
    y = np.zeros((3, 1024))
    p = np.broadcast_to(np.array([[0, 1, 0], [0, 0, 1], [0, 0, 1]])[:, :, None], (3, 3, 1024))
    cert = certify_forecasts(p, y, y, bounds=(0, 1))
    choice = select_horizon(cert, ranks=[7, 4, 2], tolerance=0.30)
    assert (choice.index, choice.rank, choice.horizon) == (2, 2, 2)
    abstain = select_horizon(cert, ranks=[7, 4, 2], tolerance=0.01)
    assert (abstain.index, abstain.rank, abstain.horizon) == (None, None, 0)
    # If only the candidate with a failed intermediate step is available, stop at 1.
    cert = certify_forecasts(p[:1], y, y, bounds=(0, 1))
    assert select_horizon(cert, ranks=[7], tolerance=0.30).horizon == 1


def test_single_outcome_confirms_improvement_without_estimating_noise():
    y = np.tile([0, 1], 4096)[None, :]
    p = np.stack([np.zeros_like(y), np.full_like(y, 0.5, dtype=float)])
    gain = certify_improvement(p, y, baseline=0, bounds=(0, 1))
    np.testing.assert_allclose(gain.estimate[:, 0], [0.0, 0.25])
    assert gain.lower[0, 0] == 0.0
    assert 0 < gain.lower[1, 0] < 0.25


@pytest.mark.parametrize("method", ["hoeffding", "empirical_bernstein"])
def test_exact_finite_binomial_coverage_of_paired_contrast(method):
    # The observations are exchangeable, so sum the exact probability over counts.
    # This checks the one-sided sign, units and variance on *all* possible samples.
    n, q, delta = 80, 0.25, 0.20
    p = np.stack([np.zeros((1, n)), np.ones((1, n))])
    true_gain = 2*q - 1
    violation = 0.0
    for k in range(n+1):
        y = (np.arange(n) < k)[None, :].astype(float)
        lower = certify_improvement(p, y, baseline=0, bounds=(0, 1), delta=delta, method=method).lower
        if lower[1, 0] > true_gain:
            violation += math.comb(n, k) * q**k * (1-q)**(n-k)
    assert 0.0 < violation <= delta


def test_invalid_calibration_inputs_fail_loudly():
    p = np.zeros((1, 1, 20))
    y = np.zeros((1, 20))
    for wrong in (np.full_like(y, 1.01), np.full_like(y, np.nan), y[:, :-1]):
        with pytest.raises(ValueError):
            certify_forecasts(p, y, wrong, bounds=(0, 1))
    with pytest.raises(ValueError):
        certify_forecasts(p, y, bounds=(0, 1), delta=1.0)
    with pytest.raises(ValueError):
        certify_forecasts(p[:, :, :1], y[:, :1], bounds=(0, 1))
    with pytest.raises(ValueError):
        certify_forecasts(p, y, bounds=(1, 1))
    with pytest.raises(ValueError):
        certify_forecasts(p, y, bounds=(0, 1), method="unknown")
    with pytest.raises(ValueError):
        certify_improvement(p, y, baseline=0, bounds=(0, 1))
    cert = certify_forecasts(p, y, bounds=(0, 1))
    for ranks, tolerance in (([], 1), ([1.5], 1), ([0], 1), ([1], -1)):
        with pytest.raises(ValueError):
            select_horizon(cert, ranks=ranks, tolerance=tolerance)


@pytest.mark.parametrize("method", ["hoeffding", "empirical_bernstein"])
def test_finite_support_diameter_and_global_extreme_equivalence(method):
    # At horizon 0 the difference is small. At horizon 1 the predictors
    # occupy opposite endpoints, recovering the worst-case diameter exactly.
    support = np.array([[[.45, .50], [0., 1.]], [[.55, .55], [1., 0.]]])
    ids = np.tile([0, 1], 512)
    p = support[:, :, ids]
    y = np.broadcast_to((ids == 0)[None], (2, len(ids))).astype(float)
    advice = dict(baseline=0, bounds=(0, 1), delta=0.05, method=method)
    generic = certify_improvement(p, y, **advice)
    tight = certify_improvement(p, y, support_predictions=support, support_indices=ids, **advice)
    np.testing.assert_array_equal(tight.estimate, generic.estimate)
    assert tight.lower[1, 0] > generic.lower[1, 0]
    np.testing.assert_array_equal(tight.lower[:, 1], generic.lower[:, 1])
    # Enumerate the extrema independently and check the derived confidence radius.
    diffs = [(q - endpoint)**2 - (z - endpoint)**2
             for q, z in zip(support[0, 0], support[1, 0]) for endpoint in (0., 1.)]
    d = max(diffs) - min(diffs)
    losses = (p[0, 0] - y[0])**2 - (p[1, 0] - y[0])**2
    n = len(ids)
    radius = (d*np.sqrt(np.log(2/.05)/(2*n)) if method == "hoeffding" else
              np.sqrt(2*np.var(losses, ddof=1)*np.log(4/.05)/n) +
              7*d*np.log(4/.05)/(3*(n-1)))
    np.testing.assert_allclose(tight.lower[1, 0], losses.mean()-radius)


def test_finite_support_rejects_inconsistent_tables_and_state_indices():
    table = np.array([[[0., 0.5]], [[0.2, 1.]]])
    ids = np.array([0, 1, 0])
    p = table[:, :, ids]
    y = np.zeros((1, len(ids)))
    for support, indices in ((table, None), (None, ids), (table[:, :, :0], ids),
                             (table, ids[::-1]+1), (table, ids.astype(float)),
                             (table[:, :, ::-1], ids), (np.full_like(table, np.nan), ids)):
        with pytest.raises(ValueError):
            certify_improvement(p, y, baseline=0, bounds=(0, 1),
                                support_predictions=support, support_indices=indices)
    # Identical deployment scores imply an identically zero paired loss difference.
    identical = np.zeros_like(table)
    cert = certify_improvement(identical[:, :, ids], y, baseline=0, bounds=(0, 1),
                               support_predictions=identical, support_indices=ids[None])
    np.testing.assert_array_equal(cert.lower, 0.)
