# Forecasting with time-indexed singular spaces: CPU pilot

All scores below are absolute conditional-mean RMSE of forecasts clipped to [−1,1]. Entries average over five horizons first, then paired training seeds. Lower is better. Analytic conditional means are used only for independent evaluation, never for fitting or selection.

| System | Regime | Pooled EDMD | Time EDMD | Direct OLS | Known-closure EDMD | Rank 3 + bridge | Rank 3, identity |
|---|---|---:|---:|---:|---:|---:|---:|
| finite chain | stationary | 0.0113 | 0.0228 | 0.0291 | 0.0228 | 0.0549 | 0.2227 |
| finite chain | transient | 0.0144 | 0.0249 | 0.0370 | 0.0249 | 0.0583 | 0.1682 |
| finite chain | switching | 0.1583 | 0.0250 | 0.0348 | 0.0250 | 0.0795 | 0.1359 |
| warped circle | stationary | 0.0169 | 0.0350 | 0.0502 | 0.0239 | 0.1763 | 0.3301 |
| warped circle | transient | 0.0170 | 0.0366 | 0.0481 | 0.0220 | 0.1573 | 0.2472 |
| warped circle | switching | 0.2557 | 0.0335 | 0.0559 | 0.0198 | 0.1313 | 0.2690 |

The rank-3 and identity columns have identical data, dictionary, singular functions and readout. Replacing each cross-basis overlap by an identity deletes basis orientation information; this is a deliberately incorrect ablation, not a competitive baseline. The pooled, time EDMD and direct OLS baselines use the same random tanh dictionary with no imposed rank-3 cap. Direct OLS fits a separate regression for each horizon. Known-closure EDMD has privileged features (tabular or four latent harmonics) but estimates its dynamics from the same paths.

## Selection and certification

The predeclared menu uses ranks [1, 3, 5, 6] on the chain and [1, 3, 5, 8] on the circle. The RMSE tolerance is 0.30. The longest acceptable contiguous prefix is selected; ties prefer smaller rank. Horizon 0 abstains. Bounds cover each complete fitted forecast, not separate uncertainty components for its factors. At each seed/regime, split the failure budget 0.05 equally between the absolute-risk and paired-improvement families. No single joint guarantee over different seeds or systems is claimed.

| System | Regime | Selected rank by seed (0 = abstain) | Certified horizon by seed | Risk family covered | Positive lower bounds vs time EDMD |
|---|---|---|---|---:|---:|
| finite chain | stationary | [3, 3, 3, 3, 3] | [5, 5, 5, 5, 5] | 5/5 | 0/100 |
| finite chain | transient | [3, 3, 3, 3, 3] | [5, 5, 5, 5, 5] | 5/5 | 0/100 |
| finite chain | switching | [3, 3, 3, 3, 3] | [5, 5, 5, 5, 5] | 5/5 | 0/100 |
| warped circle | stationary | [5, 3, 3, 8, 3] | [5, 5, 5, 5, 5] | 5/5 | 0/100 |
| warped circle | transient | [3, 3, 3, 3, 3] | [5, 5, 5, 5, 5] | 5/5 | 0/100 |
| warped circle | switching | [3, 3, 3, 3, 3] | [5, 5, 5, 5, 5] | 5/5 | 0/100 |

With the switching chain fit fixed at training seed 0, 64/64 fresh calibration experiments covered all ranks/horizons. This is a check at one fixed training realization, not a general coverage estimate.

## Design and interpretation

- For each of 5 seeds: 1024 independent training paths, 4096 independent calibration starts with two independent continuations, and 2048 independent test starts. The public JSON records every model/horizon, signed calibration estimates, MSE upper bounds, and lower bounds for baseline-minus-candidate MSE. The sample unit is a trajectory start.
- Primary metric fixed before viewing results: mean over horizons 1–5 of population RMSE; primary spectral model rank 3; primary competitive comparison is time-specific EDMD. All outputs use one bounded scalar observable. No controller, trajectory-tail or pointwise guarantee is implied.
- Finite chain: six states; each step holds with probability .30, shifts cyclically with probability .58, and draws uniform with probability .12. Exact risk sums over the starting distribution. The observable is .65 cos(theta) + .35 sin(2 theta).
- Circle: latent angle advances by a time-specific phase plus Gaussian noise (SD .18) with probability .88, or resets uniformly. The observed circle angle is theta + .45 sin(theta); the observable is .65 cos(theta) + .35 sin(3 theta). For harmonic k, the h-step multiplier is .88^h exp(−k² h .18²/2) exp(i k sum(phase)). Population MSE uses 1,024-point periodic quadrature. A second 2,048-point grid checks numerical error. A separate single-start Monte Carlo check compares each analytic horizon against 20,000 simulated continuations (per seed and case).
- Both stationary controls use a uniform starting law and fixed transitions. Transient uses a biased starting law with the identical homogeneous transition. Switching changes the transitions as well. Absolute risk is checked against the exact chain sum or the refined numerical circle integral.
- The estimator uses 16 fixed random tanh features per side; it learns the empirical linear algebra of those features from training paths. This study does not optimize neural weights or establish a neural advantage. Time indices and synchronized replicate starts are known.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python experiments/nonstationary_pilot.py --seeds 5 --train 1024 --cal 4096 --test 2048 --replicates 64 --tolerance 0.3 --delta 0.05
```
