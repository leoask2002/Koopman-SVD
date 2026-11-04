# Neural singular functions across changing dynamics

This bounded CPU experiment compares actual nested-loss neural optimization against the same networks at initialization and two tabular benchmarks. It uses a synthetic system explicitly constructed to have an exact rank-two Koopman representation.

At time t the observable f_t is the ±1 majority of three observed bits; the triple varies by time. The fifth bit is a nuisance coordinate whose positive probability changes from 0.25 to 0.70, 0.40 and 0.80. The first four bits remain uniformly distributed at each time. The transition law is P_t(y|x) = ρ_{t+1}(y)[1 + λ_t f_t(x) f_{t+1}(y)], with λ=(0.90, −0.75, 0.85). Each row sums to one and maps ρ_t to ρ_{t+1}. Consequently the exact mean of f_h given X_0=x is the product of the first h λ's times f_0(x); the script also verifies this by explicit transition-matrix multiplication.

The primary endpoint is the equal-weight mean of the three horizon-wise conditional-mean RMSEs, in units of an observable ranging from −1 to 1. Every method sees the same training paths and test starts. There are 640 training, 320 validation and 4,096 test trajectories per seed, split before pair extraction. The nonlinear target is supplied to each decoder; its analytic expectation is never used for fitting or checkpoint selection.

| Model | h=1 | h=2 | h=3 | Primary mean ± seed SD |
|---|---:|---:|---:|---:|
| full_table_edmd | 0.0844 | 0.0773 | 0.0783 | 0.0800 ± 0.0073 |
| rank2_coordinates | 0.4841 | 0.4139 | 0.4006 | 0.4329 ± 0.0120 |
| rank2_onehot | 0.0899 | 0.0805 | 0.0781 | 0.0829 ± 0.0032 |
| frozen_neural | 0.8665 | 0.6761 | 0.5757 | 0.7061 ± 0.0101 |
| nested_neural | 0.3260 | 0.2687 | 0.2492 | 0.2813 ± 0.0273 |

Optimization changes the primary RMSE from 0.7061 (frozen) to 0.2813 (nested), a -60.2% change. The stronger of the two tabular controls is full_table_edmd at 0.0800.

The two neural arms use identical rank-two, width-12 tanh MLPs, initialization, normalization, post-fit calibration and time-specific overlap matrices. The only difference is 64 Adam updates per time step (three steps per seed, 576 updates total across three seeds); checkpointing uses validation spectral loss. Rank-two coordinate and one-hot arms use the same transport estimator with fixed dictionaries. Full-table EDMD counts all 32 states, adding one empirical-marginal pseudo-observation per row to handle sparsity. SD quantifies variation across three seeds, not a confidence interval. No settings were selected on test errors.

This checks whether representation learning beats its own initialization for a known low-rank nonlinear system. The target and operator were chosen to admit an exact rank-two model, the finite state table is available, and no advantage over generic real-world systems follows. This is not a test of certificate coverage, and a forecast is a conditional mean, not a sample path.

Reproduce: `PYTHONPATH=src python experiments/neural_transport_smoke.py` (or run directly after installing the project). The JSON contains all per-seed metrics and checkpoint selections.
