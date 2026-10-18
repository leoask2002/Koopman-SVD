# Verified results and remaining limits

The implementation is operational and the mathematical direction has demonstrable potential in these finite stochastic checks. This is a reproducible exploratory CPU study, not a full training campaign or evidence of general superiority over EDMD. Values below are means ± population standard deviations across seeds 0, 1, 2.

| Quantity | Result | Meaning |
|---|---:|---|
| Raw neural top-3 Markov spectrum error | 1.30% ± 0.22% | Ordered neural factor-norm estimates before dense calibration |
| Raw neural affine-Gaussian spectrum error after refinement | 1.24% ± 0.49% | 2,000 updates of a tiny 3D affine network, batch 1,024 |
| Calibrated affine-Gaussian spectrum error | 0.89% ± 0.50% | Finite affine-subspace reference; not the full nonlinear Gaussian spectrum |
| Full-rank Markov one-step conditional-mean error | 3.23% ± 0.18% | Equal to the fitted full-state EDMD reference here |
| Triangular one-step error, standard singular-kernel predictor | 13.64% ± 1.96% | General learned rank-six dictionary |
| Triangular one-step error, optional forecasting head | 5.96% ± 2.57% | Separate state-augmented EDMD; does not improve the SVD itself |
| Logistic one-step error, standard predictor | 3.50% ± 0.46% | Long-horizon individual predictions remain poor |
| Logistic ten-step error, standard predictor | 57.56% ± 0.28% | Illustrates the limited forecast horizon |

Relative errors use RMS ground-truth magnitude. Long-horizon ground truths of contracting systems can approach zero: the detailed JSON also records absolute RMSE and normalization by initial-state RMS.

## What the comparisons establish

- The rank-three Markov model improves over the same untrained neural dictionary: one-step error averages 48.69% ± 3.20% before learning versus 34.81% ± 1.24% afterward. Its truncation reference is about 33.55% ± 0.00%; retaining rank four removes most of this compression error.
- Longer but still bounded affine optimization improves raw neural singular factors. Calibration already recovered the complete affine span before training, so calibrated affine prediction does not demonstrate a neural advantage.
- NeuralSVD optimizes spectral factorization, not coordinate forecasting. In the deterministic examples, training can worsen forecasts compared with calibrated random features. Those negative results are retained.
- A separate state-augmented forecasting head helps the triangular example, but its known exact polynomial closure gives EDMD essentially machine-precision predictions. It remains the stronger baseline on that constructed problem.
- No decaying spectrum or finite-rank approximation in operator norm is claimed for the noiseless infinite-dimensional logistic Koopman operator. See [the mathematical discussion](../docs/mathematics.md).

## Artifacts and reproduction

- [Main three-seed results](smoke/results.md) and [machine-readable values](smoke/results.json).
- [Affine neural convergence refinement](affine-refinement/results.md).
- [Full-state Markov check](full-rank/results.md).
- [Separate forecasting-head selection and results](forecast/results.md).
- [Comparison figure](smoke/comparison.png).
- [Repository audit](../docs/audit.md) and [mathematical derivation](../docs/mathematics.md).

Every report contains the exact command, versions and budgets. The default smoke runs use at most 600 updates per model; the tiny affine refinement allows at most 2,000. All fitting is CPU-only and exits synchronously. No large training job was started or left running.

## Next scientific tests

1. Evaluate fixed neural architectures and baselines on a fresh stochastic nonlinear dataset, with trajectory-separated validation/test sets and explicit observation/process noise.
2. Use held-out conditional-expectation or multi-trajectory estimators to test singular equations and subspace convergence independently of calibration.
3. Study rank, sample count, minibatch noise and approximation capacity before scaling dimensions. Preserve ordered raw-spectrum, orthogonality and predictive metrics separately.
4. Compare against the direct NeurIPS 2025 NeuralKoopmanSVD implementation and current Kooplearn estimators under matched data and compute budgets.
5. Treat controlled/continuous-time dynamics as a further project: identification alone does not provide a robust controller or a stability guarantee.
