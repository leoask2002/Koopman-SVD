# State-augmented EDMD forecasting refinement

This is a separate post-fit forecasting head, not an improvement to the NeuralSVD decomposition or its singular values.
Its dictionary is [1, observed state, calibrated learned left functions]. All scaling and regression use training transitions only.
Dictionary rank and ridge are chosen by the mean squared validation relative RMSE over horizons 1, 5, and 10.
Known synthetic dynamics provide multi-step validation trajectories. Real data would require measured held-out validation trajectories.
Independent test initial states are evaluated only after selection. No test metric selects rank, ridge, or a neural checkpoint.
The benchmark design was developed iteratively; these are exploratory synthetic results, not a confirmatory holdout study.
Relative RMSE is sqrt(mean squared prediction error / mean squared ground truth); absolute and initial-state-normalized errors are also in results.json.

| Case | Seed | Head rank | Ridge | Standard 1-step | Augmented 1-step | Polynomial 1-step | Standard 10-step | Augmented 10-step | Polynomial 10-step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| triangular | 0 | 8 | 0 | 14.592% | 2.530% | 0.000% | 233.343% | 39.275% | 0.000% |
| logistic | 0 | 7 | 0 | 4.027% | 3.378% | 0.000% | 57.304% | 57.304% | 57.298% |
| triangular | 1 | 7 | 0 | 15.417% | 6.620% | 0.000% | 250.972% | 82.126% | 0.000% |
| logistic | 1 | 6 | 0 | 2.914% | 3.012% | 0.000% | 57.957% | 57.955% | 57.986% |
| triangular | 2 | 4 | 0.001 | 10.909% | 8.716% | 0.000% | 203.229% | 69.445% | 0.000% |
| logistic | 2 | 7 | 0.001 | 3.546% | 2.891% | 0.000% | 57.426% | 57.439% | 57.427% |

## Limits and interpretation

- State augmentation changes the forecasting dictionary and its regression objective; it does not repair or re-estimate the neural singular decomposition.
- The polynomial triangular baseline contains its exact closure, so its near-zero error is expected. The unconstrained neural dictionary has no such guarantee.
- Logistic chaos limits long-horizon state prediction: low one-step error does not imply useful individual trajectories at ten steps.
- Large triangular relative errors can result from decaying ground-truth state magnitudes. Inspect the absolute and initial-RMS-normalized errors as well.
- EDMD matrix powers are unconstrained; spectral radius and numerical conditioning are reported separately. No stability guarantee is imposed.
- Three small synthetic seeds establish reproducibility, not general superiority or statistical significance.

## Complete horizon comparisons

| Case | Seed | Method | 1-step | 5-step | 10-step |
|---|---:|---|---:|---:|---:|
| triangular | 0 | standard_neural_svd | 14.592% | 72.946% | 233.343% |
| triangular | 0 | state_augmented_edmd | 2.530% | 13.472% | 39.275% |
| triangular | 0 | reference_polynomial_edmd | 0.000% | 0.000% | 0.000% |
| triangular | 0 | untrained_neural_svd | 9.440% | 45.105% | 201.371% |
| logistic | 0 | standard_neural_svd | 4.027% | 56.943% | 57.304% |
| logistic | 0 | state_augmented_edmd | 3.378% | 56.962% | 57.304% |
| logistic | 0 | reference_polynomial_edmd | 0.000% | 57.036% | 57.298% |
| logistic | 0 | untrained_neural_svd | 0.504% | 56.938% | 57.308% |
| triangular | 1 | standard_neural_svd | 15.417% | 68.421% | 250.972% |
| triangular | 1 | state_augmented_edmd | 6.620% | 26.167% | 82.126% |
| triangular | 1 | reference_polynomial_edmd | 0.000% | 0.000% | 0.000% |
| triangular | 1 | untrained_neural_svd | 12.649% | 39.819% | 115.123% |
| logistic | 1 | standard_neural_svd | 2.914% | 58.255% | 57.957% |
| logistic | 1 | state_augmented_edmd | 3.012% | 58.225% | 57.955% |
| logistic | 1 | reference_polynomial_edmd | 0.000% | 58.342% | 57.986% |
| logistic | 1 | untrained_neural_svd | 2.594% | 58.245% | 57.975% |
| triangular | 2 | standard_neural_svd | 10.909% | 56.529% | 203.229% |
| triangular | 2 | state_augmented_edmd | 8.716% | 33.074% | 69.445% |
| triangular | 2 | reference_polynomial_edmd | 0.000% | 0.000% | 0.000% |
| triangular | 2 | untrained_neural_svd | 8.668% | 35.553% | 138.446% |
| logistic | 2 | standard_neural_svd | 3.546% | 57.024% | 57.426% |
| logistic | 2 | state_augmented_edmd | 2.891% | 57.037% | 57.439% |
| logistic | 2 | reference_polynomial_edmd | 0.000% | 57.047% | 57.427% |
| logistic | 2 | untrained_neural_svd | 1.365% | 57.035% | 57.424% |

## Reproduction metadata

```json
{
  "python": "3.12.14",
  "numpy": "2.5.4",
  "torch": "2.14.1+cpu",
  "device": "cpu",
  "threads": 1,
  "seeds": [
    0,
    1,
    2
  ],
  "steps": 600,
  "n_train": 4096,
  "max_seconds_per_neural_fit": 30.0,
  "horizons": [
    1,
    5,
    10
  ],
  "ridge_grid": [
    0.0,
    1e-08,
    1e-05,
    0.001
  ],
  "design_rcond": 1e-10,
  "selection": "mean squared validation relative RMSE at horizons 1, 5, 10",
  "validation_trajectory_source": "known synthetic dynamics; no test-driven selection",
  "command": "python -m koopman_svd.experiments_refinement --output reports/forecast"
}
```
