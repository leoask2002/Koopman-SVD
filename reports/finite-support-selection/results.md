# Reducing the calibration budget with a known finite support

This experiment compares two confidence bounds for the same pair of fitted tabular rank-two candidates (continuous-valued singular scores versus training-only two-means) under the [binary and four-level systems](../partition-selection/results.md). Keep the soft model unless a positive lower confidence bound establishes that hardening improves conditional-mean MSE. Report evidence for soft separately from inconclusive decisions.

The global comparison uses the full possible range of a paired squared-loss difference. The refined comparison uses its exact range over all **32 possible initial states** and the two observable endpoints −1 and +1. The table is obtained after fitting and before calibration; no empirical extrema, true transition probabilities or conditional means are used in constructing the bound. Both use the same empirical variance, calibration draws, family budget, prediction clipping and two-sided gain definition. See the [proof and API](../../docs/certification.md).

Fresh master seed 2026110312; 64 independent refits, 640 training paths per task, and nested calibration sizes fixed at 1,024 and 8,192. Each start supplies one independent trajectory (three correlated horizons). Allocate 0.05/(2 methods × 2 tasks × 2 sizes × 2 directions) per call; each call covers all horizons. Hence all 24 two-sided intervals per trial have ≥95% simultaneous confidence. Exact risk sums over the entire initial law using conditional means independently verified by transition-matrix multiplication.

| Task | Starts | Score range | Certified hard | Certified soft | Inconclusive | Selected RMSE | Oracle RMSE |
|---|---:|---|---:|---:|---:|---:|---:|
| binary | 1,024 | global | 0.0% | 0.0% | 100.0% | 0.0854 | 0.0305 |
| binary | 1,024 | complete_support | 0.0% | 0.0% | 100.0% | 0.0854 | 0.0305 |
| binary | 8,192 | global | 0.5% | 0.0% | 99.5% | 0.0846 | 0.0305 |
| binary | 8,192 | complete_support | 31.2% | 0.0% | 68.8% | 0.0612 | 0.0305 |
| four_level | 1,024 | global | 0.0% | 0.0% | 100.0% | 0.0980 | 0.0980 |
| four_level | 1,024 | complete_support | 0.0% | 0.0% | 100.0% | 0.0980 | 0.0980 |
| four_level | 8,192 | global | 0.0% | 25.5% | 74.5% | 0.0980 | 0.0980 |
| four_level | 8,192 | complete_support | 0.0% | 41.7% | 58.3% | 0.0980 | 0.0980 |

Decision percentages pool three horizons within each fit; they are not independent sampling units. RMSE is averaged over horizons and fitted-model trials; the oracle chooses the lower true-risk candidate separately for each fit and horizon.

| Task | Always soft RMSE | Always hard RMSE | Mean support/global score diameter |
|---|---:|---:|---:|
| binary | 0.0854 | 0.0305 | 15.5% |
| four_level | 0.0980 | 0.1428 | 18.2% |

Joint intervals covered all exact differences in 64/64 trials. One-sided 95% exact binomial lower limit: 0.954.
This is finite Monte Carlo evidence for the two generators, not proof of uniform real-world coverage. Inspect the JSON for wrong-direction rates and interval widths. The tighter range only reduces the range-dependent term in the empirical-Bernstein radius; the sampling-variance term remains. Small data and small true differences can still leave most decisions unresolved. A smaller but invalid range taken from sampled extrema could appear to improve decisions while destroying coverage.

Scope: this experiment takes advantage of an exhaustive finite state alphabet. The results do not transfer automatically to continuous state spaces, uncontrolled distribution shifts, unknown support, or neural architectures. No neural weights were trained. Reproduce: `OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python experiments/finite_support_selection.py`. The JSONL contains risks and confidence endpoints for every fit, task, size and method.
