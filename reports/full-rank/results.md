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
| markov | 0 | 3.492% | 1.277% | 2.973% | 2.973% | 2.787% | 2.787% |
| markov | 1 | 2.333% | 1.234% | 3.353% | 3.353% | 5.495% | 5.495% |
| markov | 2 | 1.932% | 1.260% | 3.358% | 3.358% | 4.987% | 4.987% |

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
  "markov_rank": 4,
  "batch_size": 256,
  "command": "python -m koopman_svd.experiments --cases markov --markov-rank 4 --seeds 0 1 2 --output reports/full-rank",
  "kooplearn": "2.0.3"
}
```

## Neural training ablation

| Case | Seed | Raw neural 1-step error | Random features + calibration | Trained features + calibration | 10-step error / initial RMS |
|---|---:|---:|---:|---:|---:|
| markov | 0 | 5.625% | 2.973% | 2.973% | 1.402% |
| markov | 1 | 7.804% | 3.353% | 3.353% | 2.763% |
| markov | 2 | 6.332% | 3.358% | 3.358% | 2.508% |
