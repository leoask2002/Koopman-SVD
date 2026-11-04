# Forecasting observables under changing state distributions

The goal of this study is a usable conditional-mean forecast, not a small error in a list of singular values. Three distinct effects need their own controls: learning informative features, composing them in the right coordinates, and sharing observations across times. The [bounded pilot](../reports/nonstationary/), [neural ablation](../reports/neural-transport/), and [Gaussian reference](../reports/gaussian-transport/) isolate different effects; their RMSEs have different units and cannot be averaged across studies.

## Model and conditions

Take independent, aligned trajectories with state law $\rho_t$ at step $t$. The Markov conditional expectation $K_t:L^2(\rho_{t+1})\to L^2(\rho_t)$ maps an endpoint observable back to its expectation given the present state. Fit on complete training paths. The empirical estimator centers each feature dictionary in its own time marginal, drops unsupported Gram directions, whitens the remaining columns, and computes a small projected cross-moment SVD. It pins the constant to the first basis vector. This is a finite projected SVD. For learned neural representations, independently fit the nested-loss models first; the time-indexed estimator calibrates their frozen spans and implements the transport. For prescribed features there is no network training.

Write $\widehat K_t h=\Phi_t^\top\Sigma_t\langle\Psi_t,h\rangle_{\rho_{t+1}}$. The basis change needed for two consecutive factors is
```math
B_t=\mathbb E_{Z\sim\rho_{t+1}}[\Psi_t(Z)\Phi_{t+1}(Z)^\top].
```
Use both functions on the **same intermediate state**. For $s<u$, a forecast of $h(X_u)$ has coefficients $\Sigma_s B_s\Sigma_{s+1}\cdots B_{u-2}\Sigma_{u-1}\langle\Psi_{u-1},h\rangle_{\rho_u}$ in the left basis at time $s$. This covers different ranks per step. Keeping an identity bridge or raising the singular values to a power generally fails. An identity bridge is also sensitive to arbitrary sign conventions. On the matched empirical distributions the bases are orthonormal; Cauchy–Schwarz bounds the singular-step and bridge norms by one in those empirical spaces. This does not imply positivity, pointwise stability off distribution, or an accuracy certificate on all observables.

Separate changing laws from changing transition rules. For a homogeneous forward matrix $P$, the reverse conditional at step $t$ is $Q_t=D_{\rho_{t+1}}^{-1}P^\top D_{\rho_t}$ on supported states. The forward two-step operator is still $P^2$; the reverse product is $Q_1 Q_0$. With rows $(.9,.1),(.2,.8)$ and initial law $(.8,.2)$, $\max_{ij}|(Q_1Q_0-Q_0^2)_{ij}|=0.030549$. This elementary calculation motivates the changing-law controls; it is not by itself a research contribution. When forward dynamics are homogeneous, repeated estimation at every time can be inferior to pooling.

## Working API

The following example assumes `train` is shaped (trajectory, time, state); `x0` contains independent calibration starts from the deployment start law; and `y1,y2` are scalar bounded outcomes from two independent continuations of each start, shaped (horizon, start). No checkpoint, dictionary, bridge, or candidate menu may be selected using these outcomes prior to certification. Use the same clipping at deployment.

```python
import numpy as np
from koopman_svd import TimeIndexedKoopmanSVD, certify_forecasts, select_horizon

# Use fitted dictionaries in practice; this is a fixed nonlinear feature baseline.
def features(x):
    return np.column_stack((x, np.tanh(2.0 * x)))

def observable(x):
    return np.tanh(x[:, 0])  # guaranteed range [-1, 1]

ranks = (1, 2, 3)
max_model = TimeIndexedKoopmanSVD.fit(train, rank=max(ranks),
                                       left_features=features, right_features=features)
models = [max_model.truncate(r) for r in ranks]
pred = np.stack([[m.predict(x0, horizon=h, observable=observable)[:, 0]
                  for h in range(1, len(y1) + 1)] for m in models])
cert = certify_forecasts(pred, y1, y2, bounds=(-1, 1), delta=0.05)
selected = select_horizon(cert, ranks=ranks, tolerance=0.15)
# selected.horizon == 0 means abstain; otherwise certify the initial contiguous prefix.
# For any selected model/horizon, deploy np.clip(prediction, -1, 1).
```

For neural factors use `TimeIndexedKoopmanSVD.from_neural_models(train, fitted_models, rank=...)`, where model t was fitted on transitions at t (or a frozen shared dictionary is repeated). The [runnable example](../experiments/neural_transport_smoke.py) uses matched initializations and an independent validation split. The fitted objects referenced by a transport estimator must remain frozen. All moments for composition and the endpoint readout come from training trajectories. Reusing those same paths to compute a claimed validation risk would be invalid.

## What the experiment validates

The [certificate derivation](certification.md) covers the final clipped scalar predictor as a whole. It does **not** bound each whitening or bridge operator separately. Replicated trajectories identify conditional-mean squared risk via a product of residuals. One ordinary trajectory per start can support an upper bound including process noise, or a paired **risk difference** relative to a fixed comparator. Confidence is over calibration sampling conditional on the independently built candidate family. Varying model seeds and repeating calibration are different experiments. A nominal guarantee is uninformative if the bound is too wide to accept any useful horizon. Where the initial state support is known and finite, the [paired comparison](../reports/finite-support-selection/results.md) can use exact extrema of the gain over the complete support to tighten its range term. The implementation checks lookup consistency, but cannot establish support completeness from the calibration observations.

## What would merit a larger study

1. **Separate drift sources before increasing model size.** Where a time-homogeneous transition fits, estimate it from pooled data and update marginal-dependent reversal. Where the transition changes, compare local, partially pooled, time-neighbor smoothed, shared-feature and direct-horizon fits. Select on independent trajectories. Report the strongest baseline, even if it wins.
2. **Ask whether neural compression pays for itself.** Use a higher-dimensional observed system with a known small latent stochastic system and held-out observation maps. Keep the transition counts, optimizer budgets and validation access fixed. Compare learned features, frozen features, structural dictionaries, full/ridge EDMD where feasible, and the authors’ [NeuralKoopmanSVD implementation](https://github.com/MinchanJeong/NeuralKoopmanSVD). Test on new seeds and system parameters, not merely new draws from one hand-built chain.
3. **Measure utility before calling a bound useful.** Fix the observable, start distribution, tolerance, primary horizon, candidate ranks and calibration budget in advance. Report the accepted horizon (including abstentions), actual conditional-mean risk, false acceptances, calibration cost and baseline-relative risk. Replication requires resets or a simulator; test a paired single-successor advantage first where resets are impossible. A single dependent stream needs new statistical assumptions; do not count its overlapping windows as independent starts.
4. **Gate for scaling.** Advance to application data if improvements survive a matched strong comparator or a well-defined compression/compute constraint on a fresh system, and the bounds accept a nontrivial horizon at the operational tolerance. Otherwise keep the estimator as a diagnostic or use the simpler forecaster. No closed-loop stability conclusion follows from these tests.

## Research context

- [Ryu et al., ICML 2024](https://proceedings.mlr.press/v235/ryu24b.html) and [Jeong et al., NeurIPS 2025](https://arxiv.org/abs/2507.07222): nested neural SVD and stochastic Koopman singular functions, respectively.
- [Wu–Noé](https://arxiv.org/abs/1707.04659) already treats distinct marginals and nonstationary data; [Turri et al., ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/6d55265321be68a6a346b082eafb593b-Paper-Conference.pdf) relates evolution operator learning to self-supervised representations.
- [Kostic et al., ICML 2024](https://proceedings.mlr.press/v235/kostic24a.html), [Philipp et al., J. Nonlinear Science 2025](https://doi.org/10.1007/s00332-025-10182-3), and [Conradie et al., 2026 preprint](https://arxiv.org/abs/2603.15091) address long-horizon forecasting, non-invariant sampling, and a posteriori invariance diagnostics, respectively.
- [Colbrook et al., Nonlinear Dynamics](https://doi.org/10.1007/s11071-023-09135-w) use batched snapshots to separate residuals and process variance. [Maurer–Pontil](https://arxiv.org/abs/0907.3740) give the empirical Bernstein inequality used here.
- [Xu et al., Oct. 29, 2026 preprint](https://arxiv.org/abs/2610.39580) study a two-stage operator/Galerkin estimator and error propagation in a time-homogeneous setting. [Li–Bu, 2026 preprint](https://arxiv.org/abs/2609.17815) combine learned singular functions and Kalman inference; [Kozdoba–Mannor, 2026 preprint](https://arxiv.org/abs/2608.18918) show how leading-mode truncation can hide a component. No claim of first use or general superiority is made here.
