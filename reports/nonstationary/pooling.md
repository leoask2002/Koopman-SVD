# When should transition data be pooled?

The first pilot showed opposite winners when only the initial distribution changes and when the transition law itself changes. This separately designed follow-up tests whether calibration data can distinguish them. It uses fresh seeds starting at 100; systems, observables and horizon range are unchanged. This remains an exploratory synthetic experiment.

The method blends the clipped pooled EDMD forecast (weight 0) and the clipped full available time-specific singular-function forecast (weight 1); the latter uses the same fixed tanh dictionary and the correct cross-time bridges. Choose one weight from 9 evenly spaced candidates by minimizing noisy mean squared error across all five horizons on 4096 independent calibration paths after fitting both endpoints on the same 1024 training paths. The estimator uses neither exact conditional means nor a regime label. It does use a known time index for the local endpoint and needs validation rollouts from the deployment distribution. The selected weight is fixed across horizons. This is a forecasting ensemble; its weights are not singular values, and it does not improve the SVD itself.

| System | Regime | Pooled | Time SVD | Time EDMD | Adaptive | Selected time weights (seed order) | Certified gains vs pool / vs time EDMD (of seed × horizon cases) |
|---|---|---:|---:|---:|---:|---|---|
| finite chain | stationary | 0.0128 | 0.0207 | 0.0207 | 0.0125 | [0.0, 0.25, 0.25, 0.0, 0.375] | 0 / 0 of 25 |
| finite chain | transient | 0.0151 | 0.0207 | 0.0207 | 0.0149 | [0.0, 0.375, 0.25, 0.0, 0.0] | 0 / 0 of 25 |
| finite chain | switching | 0.1579 | 0.0182 | 0.0182 | 0.0182 | [1.0, 1.0, 1.0, 1.0, 1.0] | 10 / 0 of 25 |
| warped circle | stationary | 0.0175 | 0.0349 | 0.0348 | 0.0174 | [0.0, 0.25, 0.0, 0.0, 0.0] | 0 / 0 of 25 |
| warped circle | transient | 0.0156 | 0.0335 | 0.0342 | 0.0157 | [0.0, 0.125, 0.0, 0.125, 0.125] | 0 / 0 of 25 |
| warped circle | switching | 0.2549 | 0.0326 | 0.0325 | 0.0326 | [1.0, 1.0, 1.0, 1.0, 1.0] | 17 / 0 of 25 |

Errors are conditional-mean RMSE averaged over horizons 1–5 and seeds, in bounded observable units. A positive simultaneous lower bound means the selected model improves the observable MSE at that horizon; an inconclusive bound is not a declaration of equality. Two predeclared baseline comparisons split the 5% failure budget equally within each run. Their finite menus include all weights, so selection using the same calibration outcomes is allowed.

Both full comparison families covered their population contrasts in 30/30 runs. Coverage is a diagnostic, not evidence of universal superiority. Exact finite-state risks are summed and circle risks use the periodic quadrature defined in the first pilot. There is no global guarantee over the full collection of runs.

Forecast mixing is a conventional bias–variance choice. It can reduce the loss of making the wrong pooling decision; it can also have higher error than an endpoint that is appropriate for a particular regime. The menu and fresh seed family were fixed after examining the first pilot. These experiments do not establish novelty, robust controller performance or benefit on real data.

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python experiments/adaptive_pooling.py --seeds 5 --train 1024 --cal 4096
```
