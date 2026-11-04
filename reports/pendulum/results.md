# Stochastic pendulum: unseen damping schedules and sensor changes

This is a bounded physical **simulation**, not a hardware experiment. The forecasts target the conditional mean of cos(angle). The simulator defines a discrete process; no inference about the exact continuous-time generator or closed-loop control is made. See the [fixed protocol](protocol.md) and [machine-readable results](results.json).

## Primary result

For each fit we selected one candidate in each family by horizon-six loss on independent validation schedules. Primary scores are the mean of per-seed RMSEs on full sensor A at 1.5 time units: each seed averages corrected MSE equally over two genuinely held-out damping schedules before taking a square root. Lower is better.

| Validation-selected family | Primary RMSE | Warm median ms / 128 starts | Propagated/model dimensions by fit |
|---|---:|---:|---|
| classical | 0.15850 | 0.446 | [25, 25, 25, 25, 25] |
| frozen | 0.46983 | 0.154 | [8, 8, 8, 8, 8] |
| learned | 0.47935 | 0.137 | [8, 8, 8, 4, 4] |
| privileged | 0.00486 | 1.600 | [5, 5, 5, 5, 5] |
| upstream | 0.46920 | 0.184 | [8, 8, 8, 4, 8] |

Learned minus classical: **+0.32084** (paired 95% seed-bootstrap interval [+0.30244, +0.33904]); RMSE ratio **3.024** [2.751, 3.259]. Intervals are descriptive only: 5 independent fits, two specific unseen schedules, common evaluation draws across candidates. The comparison uses validation winners, not a posteriori test winners.

## Stress tests and certificates

Entries are mean horizon-six corrected RMSE across training seeds. The prefix is the mean count of consecutive certified horizons for the validation-selected learned model at absolute RMSE tolerance 0.15. A positive gain lower bound compares learned against the validation-selected classical model at horizon six; absence of a positive bound is inconclusive.

| Observation | Test damping | Classical | Learned | Frozen | Official | Privileged | Learned prefix / 6 | Positive gain bounds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| full A | test_smooth | 0.1585 | 0.4586 | 0.4503 | 0.4505 | 0.0056 | 0.0 | 0/5 |
| full A | test_switch | 0.1568 | 0.4990 | 0.4884 | 0.4871 | 0.0041 | 0.0 | 0/5 |
| full B | test_smooth | 1.4177 | 0.8697 | 0.8801 | 0.9757 | 0.0056 | 0.0 | 4/5 |
| full B | test_switch | 1.4044 | 0.8652 | 0.8933 | 0.9582 | 0.0041 | 0.0 | 5/5 |
| partial A | test_smooth | 0.1667 | 0.1617 | 0.1655 | 0.1609 | — | 0.0 | 0/5 |
| partial A | test_switch | 0.1683 | 0.1505 | 0.1597 | 0.1490 | — | 0.0 | 0/5 |

On held-out nuisance map B the fixed harmonic EDMD control, which uses the unchanged physical anchors, scored 0.2117/0.2095 on smooth/switch. Even persistence scored 0.6384/0.6852. A positive gain over a model selected on A but failing on B is therefore not a demonstration of sensor robustness.

## Fixed candidate inspection

Every training-map-A candidate below is scored at horizon six on the two primary schedules, whether or not validation selected it. Selection counts use only the validation schedules and only competitors within the same family.

| Candidate | Family | Primary RMSE | Selected fits | Coordinates | Feature width | Warm ms / 128 |
|---|---|---:|---:|---:|---:|---:|
| persistence | classical | 0.66224 | 0/5 | 1 | 1 | 0.015 |
| edmd_pooled | classical | 0.20079 | 0/5 | 25 | 25 | 0.072 |
| edmd_smoothed | classical | 0.15850 | 5/5 | 25 | 25 | 0.446 |
| edmd_damping | classical | 0.16145 | 0/5 | 25 | 25 | 0.376 |
| direct_horizon | classical | 0.19636 | 0/5 | 25 | 25 | 0.296 |
| edmd_harmonic | classical | 0.21078 | 0/5 | 18 | 18 | 0.356 |
| edmd_rank4 | classical | 0.52925 | 0/5 | 4 | 4 | 0.239 |
| edmd_rank8 | classical | 0.43349 | 0/5 | 8 | 8 | 0.245 |
| frozen_r4 | frozen | 0.55460 | 0/5 | 4 | 4 | 0.150 |
| neural_r4 | learned | 0.50113 | 2/5 | 4 | 4 | 0.136 |
| sensor_spectral_r4 | classical | 0.47411 | 0/5 | 4 | 9 | 0.017 |
| upstream_cca_r4 | upstream | 0.50434 | 0/5 | 4 | 4 | 0.245 |
| upstream_edmd_r4 | upstream | 0.50759 | 0/5 | 4 | 4 | 0.208 |
| upstream_prefix_r4 | upstream | 0.49931 | 1/5 | 4 | 4 | 0.089 |
| frozen_r8 | frozen | 0.46983 | 5/5 | 8 | 8 | 0.154 |
| neural_r8 | learned | 0.47760 | 3/5 | 8 | 8 | 0.137 |
| sensor_spectral_r8 | classical | 0.46932 | 0/5 | 8 | 9 | 0.017 |
| upstream_cca_r8 | upstream | 0.48899 | 0/5 | 8 | 8 | 0.241 |
| upstream_edmd_r8 | upstream | 0.47528 | 3/5 | 8 | 8 | 0.199 |
| upstream_prefix_r8 | upstream | 0.48052 | 1/5 | 8 | 8 | 0.087 |
| privileged_mechanics | privileged | 0.00486 | 5/5 | 5 | 0 | 1.600 |

On the primary endpoint the average finite-reference variance contribution to raw MSE was 0.000215703 (equivalent RMSE scale 0.01469). The mean independent-test-start standard error of the corrected MSE for the selected classical model was 0.00360927. This is simulation uncertainty, not across-fit uncertainty. Unlike trajectory RMSE against single noisy endpoints, the corrected target measures conditional-mean forecasting error.

A→B changes only five redundant nuisance sensor channels. Three calibrated informative anchors are unchanged; there is no fitting or validation on B. The partial view has no velocity channel; the latent initial velocity is redrawn independently on each reference/calibration branch, so the reference conditions only on available information. The mechanics comparator is privileged (decodes the calibrated channels and assumes the governing form). Direct regression dimensions describe a dictionary rather than propagated rank; the compression gate does not treat a direct winner as a larger recurrent state.

## Cost, evidence and decision

Gate: accuracy (>10% lower primary RMSE) **False**; compression (≤5% excess, ≥2× rank reduction and faster forecasting) **False**; all six horizons certified on both primary schedules for all fits **False**. Overall advance-to-scale decision: **False**.

Per-fit model dimensions and warm-start latencies (median of five one-thread runs for 128 starts and all six steps) are reported, along with first-query timings, fit-group times, candidate-specific storage proxies and trainable parameters where defined. Model size, number of features and runtime differ; timing depends on the machine and cached coefficient availability. No large model was trained.

Per seed: 4×256 training and 2×256 validation paths; 4096 new calibration starts and 512 independent test starts with 128 independent continuations per test start/schedule. Signed variance-corrected MSE, its independent-start standard error, raw MSE, and finite-reference variance contribution are recorded for each horizon. Clipping negative corrected MSE to zero for the displayed RMSE can matter for very accurate predictors. The absolute certificates and one-sided gain bounds split a 5% familywise budget over six scenarios and two types of bound per fitted seed. They cover average risk for the stated deployment starts; they do not certify individual trajectories under arbitrary shifts.

## Reproduction

From the repository root, install the package and test extra. The full study also needs the [official adapter instructions](../../docs/pendulum-upstream.md). Then run:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python experiments/pendulum_study.py
```

The main JSON records all per-seed candidate metrics, validation choices, budgets and schedules. Optional scratch audit arrays contain individual calibration forecasts and independent test references; they are not needed to read the report. No method may use them for further tuning and still claim these are fresh confirmatory tests.

## Additional diagnostics

The near-zero mechanics score needs more conditional branches than the generic forecasts. See the [independent high-precision audit](mechanics.md) and [discretization checks](physics.md). Warm latency is a property of these implementations and their caching policies; it is not an inherent advantage of NeuralSVD. The full-A risk family uses 21 candidates and six horizons. Its empirical Bernstein range term alone adds approximately 0.0314 in squared-error units at 4,096 starts, larger than the target tolerance squared (0.0225); signed residual means and sample variances determine the actual bound. This explains why even a good forecaster can have an uninformative guarantee here.
