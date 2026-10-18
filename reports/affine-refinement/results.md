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
| gaussian_affine | 0 | 1.818% | 1.519% | 4.146% | 4.146% | 104.859% | 104.859% |
| gaussian_affine | 1 | 1.297% | 0.875% | 4.026% | 4.026% | 349.146% | 349.146% |
| gaussian_affine | 2 | 0.617% | 0.287% | 5.296% | 5.296% | 204.927% | 204.927% |

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
  "steps": 2000,
  "n_train": 4096,
  "max_seconds_per_model": 30.0,
  "markov_rank": 3,
  "batch_size": 1024,
  "command": "python -m koopman_svd.experiments --cases gaussian_affine --steps 2000 --batch-size 1024 --seeds 0 1 2 --output reports/affine-refinement",
  "kooplearn": "2.0.3"
}
```

## Neural training ablation

| Case | Seed | Raw neural 1-step error | Random features + calibration | Trained features + calibration | 10-step error / initial RMS |
|---|---:|---:|---:|---:|---:|
| gaussian_affine | 0 | 6.370% | 4.146% | 4.146% | 1.351% |
| gaussian_affine | 1 | 4.809% | 4.026% | 4.026% | 4.504% |
| gaussian_affine | 2 | 6.225% | 5.296% | 5.296% | 2.638% |
