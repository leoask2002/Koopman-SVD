# Temporal smoothing: exploratory stronger baseline

[Timing and protocol](protocol.md) · [code](../../experiments/gaussian_smoothing.py) · [per-seed results](smoothing.json).
Reproduce: `python experiments/gaussian_smoothing.py` from the repository root.

After the primary Gaussian study, we evaluated a second class of time-dependent Gaussian models: fit each affine transition by weighting nearby training times with a Gaussian kernel. Select the global bandwidth on validation trajectories. Compare to validation-selected shrinkage of time-local fits toward one global fit. Both use exactly the same data, independent validation, and time-aware Gaussian reversal.

| Regime | Pooled | Local | Global/local shrinkage | Time-neighbor smoothing | Smoothing − shrinkage (paired 95% interval) |
|---|---:|---:|---:|---:|---:|
| stationary | 0.0554 | 0.1067 | 0.0564 | 0.0568 | +0.00041 [-0.00075, +0.00171] |
| shifted_homogeneous | 0.0768 | 0.1282 | 0.0767 | 0.0777 | +0.00097 [-0.00008, +0.00245] |
| drifting | 0.2437 | 0.1371 | 0.1262 | 0.1198 | -0.00638 [-0.00953, -0.00313] |

RMSE is exact population three-step conditional-mean error as in [the primary results](README.md). The intervals describe variation over 30 independent training/validation seeds of these synthetic laws. This exploratory Gaussian comparison is conditional on smoothly varying coefficients; sharp switching, unobserved time or irregular sampling deserve separate tests. Neither method is evidence of a neural advantage. The bounded-output risk certificate elsewhere in this project does not cover Gaussian coordinates.
