# Partial pooling for changing Gaussian state distributions

[Protocol](protocol.md) · [per-seed results](results.json) · [code](../../experiments/gaussian_transport.py)

See also the [exploratory time-neighbor comparison](smoothing.md). Run `python experiments/gaussian_transport.py` from the repository root (NumPy only). No neural network was trained. This is an exact affine reference, not evidence of a neural advantage.

## Results

Entries are mean **three-step conditional-mean RMSE** across independent fits, in common synthetic state units. Each seed is scored by exact population integration across both directions, coordinates and all time origins. Lower is better. This is not RMSE against noisy observations.

| System | Fixed reverse | Pooled, changing reverse | Local | Fixed half | Endpoint choice | Adaptive |
|---|---:|---:|---:|---:|---:|---:|
| stationary | 0.0541 | 0.0554 | 0.1067 | 0.0717 | 0.0554 | 0.0564 |
| shifted_homogeneous | 0.2834 | 0.0768 | 0.1282 | 0.0918 | 0.0768 | 0.0767 |
| drifting | 0.3526 | 0.2437 | 0.1371 | 0.1454 | 0.1371 | 0.1262 |

Paired differences below are adaptive minus comparator, drifting system at h=3, the prespecified primary endpoint. Intervals are descriptive 95% percentile bootstrap intervals over 30 independent fitting seeds (10,000 resamples); they do not establish transfer to other systems or correct for multiple comparisons.

| Comparator | Paired mean RMSE difference | Interval |
|---|---:|---:|
| pooled | -0.11749 | [-0.12469, -0.11071] |
| local | -0.01090 | [-0.01494, -0.00711] |
| half | -0.01917 | [-0.02286, -0.01560] |
| endpoint_choice | -0.01090 | [-0.01494, -0.00711] |
| fixed_reverse | -0.22641 | [-0.23570, -0.21751] |

Selection frequencies (columns are weights on the local fit):

| System | 0 | 0.25 | 0.5 | 0.75 | 1 |
|---|---:|---:|---:|---:|---:|
| stationary | 21 | 9 | 0 | 0 | 0 |
| shifted_homogeneous | 21 | 9 | 0 | 0 | 0 |
| drifting | 0 | 0 | 1 | 19 | 10 |

## Estimator and interpretation

For each step, least-squares estimates of the affine forward map are shrunk toward one pooled affine fit. The same weight blends time-local residual second moments with their pooled mean. An initial sample mean and covariance are propagated through this fitted Gaussian model. The reverse conditional slope at time t is C_t A_tᵀ C_{t+1}⁻¹, with intercept m_t − B_t m_{t+1}. Reversing an interval composes these slopes in reverse time order. Each candidate is fit on training trajectories; the validation score uses observed pairs, never the analytic reference. Exact truth is used only for evaluation.

In a time-homogeneous system the forward model can share all parameters while its correct reverse map changes as the state distribution evolves. The fixed reverse diagnostic reuses a pooled backwards regression and should therefore not be the main comparator. When dynamics drift, local fits avoid homogeneous model bias but can have high estimation variance; the comparison against endpoint selection isolates the value of intermediate shrinkage weights. Fixed half-pooling is a strong control for whether validation adds value. Adaptive pooling is not novel by itself, and the preferred weight is sensitive to sample size, drift shape and prediction tasks.

All three regimes are exactly closed in the constant-plus-linear observable space. An affine Koopman model at full rank can represent the conditional means without neural approximation; the benchmark tests estimation, time reversal and pooling, not the full infinite-dimensional singular spectrum. The Gaussian observables are unbounded, so the bounded-observable coverage theorem from the other pilot does not apply.

See the JSON for forward and reverse results at horizons 1, 3, and 6 plus every seed, all validation scores, and prespecified design constants. No generator identification, single-trajectory mixing theorem, nonlinear generalization result, or controller claim is inferred.
