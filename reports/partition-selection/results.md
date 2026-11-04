# Calibration chooses when to impose two predictive classes

A finite-sample selector compares the **same fitted** time-dependent rank-two tabular model before and after partitioning its first nonconstant left/right singular scores with training-only two-means. The rule switches to two-level factors only if a paired loss-difference lower bound is positive; otherwise it retains soft factors. A negative upper bound provides evidence that hardening is harmful; a straddling interval is explicitly inconclusive. No neural networks were trained in this experiment.

## Design and exact risk

Two 32-state time-varying systems share the setup of the [neural comparison](../neural-transport/results.md): changing marginals ρ_t and transitions P_t(y|x)=ρ_{t+1}(y)[1+λ_t f_t(x) f_{t+1}(y)]. In the binary system f_t is a majority-of-three sign. In the four-level system multiply the same sign by 1 when the fifth bit is positive and 0.15 otherwise. In both cases Eρ_t[f_t]=0. The exact h-step conditional mean of the bounded outcome f_h is f_0 times the product of λ_t Eρ_{t+1}[f_{t+1}²]; multiplication of enumerated transition matrices independently verifies the formula. Evaluate each clipped predictor against this mean over all 32 initial states weighted by ρ_0 (no test Monte Carlo error).

Each fresh independent trial fits both candidates on 640 training paths for each task. Independent calibration starts each supply one stochastic continuation. Endpoint noise cancels in the population difference of squared losses against that same outcome. Run the [empirical-Bernstein comparison](../../docs/certification.md) in both directions. Allocate 0.05/(2 tasks × 2 directions × 3 fixed sizes) per call; each call simultaneously handles the three horizons. The resulting 95% family covers all 18 intervals per trial. Calibration sizes are prespecified nested prefixes. Predeclare all three sizes—no unlimited sampling until significance. Both candidates are clipped to [−1,1] at deployment, certification and exact-risk evaluation.

| Task | Calibration starts | Certified hard | Certified soft | Inconclusive | Selected RMSE | Oracle RMSE |
|---|---:|---:|---:|---:|---:|---:|
| binary | 1,024 | 0.0% | 0.0% | 100.0% | 0.0859 | 0.0326 |
| binary | 8,192 | 0.5% | 0.0% | 99.5% | 0.0852 | 0.0326 |
| binary | 65,536 | 82.3% | 0.0% | 17.7% | 0.0376 | 0.0326 |
| four_level | 1,024 | 0.0% | 0.0% | 100.0% | 0.0995 | 0.0995 |
| four_level | 8,192 | 0.0% | 27.1% | 72.9% | 0.0995 | 0.0995 |
| four_level | 65,536 | 0.0% | 57.8% | 42.2% | 0.0995 | 0.0995 |

Percentages pool the three horizons across fitted-model trials; these horizons are not independent. RMSE entries average per-horizon, per-fit square roots of exact population MSE. Oracle means whichever of the two fitted candidates has lower exact risk at that horizon; the selector never sees that information. Inconclusive comparisons default to soft, so an inconclusive interval can coincide with the oracle.

| Task | Always soft RMSE | Always hard RMSE |
|---|---:|---:|
| binary | 0.0859 | 0.0326 |
| four_level | 0.0995 | 0.1430 |

All 18 intervals jointly contained the true differences in 64/64 independent trials. The one-sided 95% binomial lower confidence limit on this generator's joint coverage probability is 0.954.
This is a stress test, not a substitute for the concentration argument. At smaller sizes the bound can be valid and still offer no decision. The experiment tests binary versus four-level finite modes on one generator family. It does not establish generic validity on a single dependent rollout, under an arbitrary deployment shift, for unbounded observables, or for neural models in particular.

Reproduce: `OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python experiments/partition_selection.py`. The JSON records the population risks, confidence intervals and per-horizon decisions for every independent trial, size and task.
