# Bounded CPU experiment results

These are small synthetic checks, not a full training campaign or a reproduction of the published benchmarks.
Every dataset uses independent training, validation and test draws; test data never choose a checkpoint.
These benchmark designs were explored iteratively; they are exploratory results, not a pristine confirmatory holdout study.
Relative RMSE is sqrt(mean squared prediction error / mean squared ground truth).
Stochastic forecasts are scored against exact conditional expectations, not individual noisy outcomes.
The Gaussian reference is restricted to affine observables. Deterministic cases report prediction only.
Post-training calibration performs a small dense SVD; the training objective itself does not.

| Case | Seed | Raw spectrum error | Calibrated spectrum error | Neural 1-step | EDMD 1-step | Neural 10-step | EDMD 10-step |
|---|---:|---:|---:|---:|---:|---:|---:|
| markov | 0 | 1.065% | 2.893% | 36.559% | 2.973% | 3.168% | 2.787% |
| gaussian_affine | 0 | 13.214% | 1.519% | 4.146% | 4.146% | 104.859% | 104.859% |
| triangular | 0 | — | — | 14.592% | 0.000% | 233.343% | 0.000% |
| logistic | 0 | — | — | 4.027% | 0.000% | 57.304% | 57.298% |
| markov | 1 | 1.601% | 1.260% | 33.986% | 3.353% | 5.666% | 5.495% |
| gaussian_affine | 1 | 6.161% | 0.875% | 4.026% | 4.026% | 349.146% | 349.146% |
| triangular | 1 | — | — | 15.417% | 0.000% | 250.972% | 0.000% |
| logistic | 1 | — | — | 2.914% | 0.000% | 57.957% | 57.986% |
| markov | 2 | 1.241% | 1.330% | 33.882% | 3.358% | 1.136% | 4.987% |
| gaussian_affine | 2 | 23.600% | 0.287% | 5.296% | 5.296% | 204.927% | 204.927% |
| triangular | 2 | — | — | 10.909% | 0.000% | 203.229% | 0.000% |
| logistic | 2 | — | — | 3.546% | 0.000% | 57.426% | 57.427% |

## Interpretation

- Long-horizon relative errors can be large when the true conditional mean decays toward zero. Absolute RMSE and error normalized by the initial-state RMS are also recorded in results.json.
- The random-feature baseline has identical seeded networks with zero optimization steps and the same post-fit calibration. This isolates learned-subspace gains from post-fit linear algebra.
- The affine case is an orientation/calibration sanity check. A complete affine span makes calibrated prediction an EDMD-type solution; it cannot establish a benefit from neural optimization.
- A rank-three Markov model compresses a four-state chain; a rank truncation and sampling floor remain. The exact truncated reference is recorded in JSON. With --markov-rank 4 the full state space is retained.
- Raw norm-based spectra need near-orthogonal converged factors. Calibrated spectra characterize the learned subspaces and can improve even when raw optimization remains incomplete.
- Triangular polynomial EDMD has a known exact closure for the observed state. The general MLP is deliberately not given that closure.
- Chaotic logistic forecasts and infinite-dimensional deterministic singular spectra have different limitations. Good one-step error does not guarantee long-horizon accuracy.
- One or a few seeds do not establish statistical significance. No large datasets, GPU runs, control policies or continuous-time generator identification are validated.

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
  "max_seconds_per_model": 30.0,
  "markov_rank": 3,
  "batch_size": 256,
  "command": "python -m koopman_svd.experiments --seeds 0 1 2 --output reports/smoke",
  "kooplearn": "2.0.3"
}
```

## Neural training ablation

| Case | Seed | Raw neural 1-step error | Random features + calibration | Trained features + calibration | 10-step error / initial RMS |
|---|---:|---:|---:|---:|---:|
| markov | 0 | 36.702% | 52.270% | 36.559% | 1.593% |
| gaussian_affine | 0 | 10.349% | 4.146% | 4.146% | 1.351% |
| triangular | 0 | 19.621% | 9.440% | 14.592% | 18.506% |
| logistic | 0 | 8.774% | 0.504% | 4.027% | 57.954% |
| markov | 1 | 34.075% | 44.508% | 33.986% | 2.849% |
| gaussian_affine | 1 | 8.300% | 4.026% | 4.026% | 4.504% |
| triangular | 1 | 19.383% | 12.649% | 15.417% | 19.708% |
| logistic | 1 | 7.167% | 2.594% | 2.914% | 57.416% |
| markov | 2 | 34.015% | 49.300% | 33.882% | 0.571% |
| gaussian_affine | 2 | 10.594% | 5.296% | 5.296% | 2.638% |
| triangular | 2 | 14.755% | 8.668% | 10.909% | 15.999% |
| logistic | 2 | 22.671% | 1.365% | 3.546% | 57.095% |
