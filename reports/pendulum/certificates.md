# What can 4,096 independent starts certify on a continuous state space?

Secondary checks on the same held-out full-map-A forecasts and stochastic outcomes. Each case uses 8,192 continuations (two per start). All models, budgets, threshold and caps were fixed before this analysis; see the [protocol](capped-protocol.md), [implementations and proofs](../../docs/certification.md), and [per-case evidence](certificates.json). This does not replace the primary scale-up gate.

## Fixed RMSE tolerance of 0.15

| Forecast | Empirical Bernstein: mean prefix | Betting: mean prefix | All six steps accepted: EB / betting (of 10 fits × schedules) |
|---|---:|---:|---:|
| persistence | 0.0 | 0.0 | 0 / 0 |
| classical | 0.0 | 4.2 | 0 / 0 |
| learned | 0.0 | 0.0 | 0 / 0 |

The procedures use identical samples, risk targets and finite candidate menus. They split a separate auxiliary 5% error allocation equally across the two schedules and two procedures of each fitted seed. The betting test answers only the single fixed threshold. It cannot be used to report an arbitrary tolerance interval without additional analysis.

## Capped observable updates

| Baseline → proposal | Maximum change | Positive lower bounds: generic / capped (of 60) | Mean h6 RMSE of capped forecast |
|---|---:|---:|---:|
| persistence -> classical | 0.10 | 43 / 60 | 0.5941 |
| persistence -> classical | 0.30 | 50 / 60 | 0.4754 |
| persistence -> classical | 2.00 | 50 / 50 | 0.1576 |
| classical -> learned | 0.10 | 0 / 0 | 0.1864 |
| classical -> learned | 0.30 | 0 / 0 | 0.2901 |
| classical -> learned | 2.00 | 0 / 0 | 0.4788 |

Each cell certifies the **capped** prediction actually deployed. A full-range cap leaves the original prediction unchanged. Smaller caps reduce the largest possible score difference, but can reduce the gain as well. Positive bounds count model-fit × schedule × horizon cells; these are correlated, not independent replications. The two pairs, two schedules and two bound methods split a separate per-seed 5% family budget; caps and horizons are simultaneous in each call. Finite-support or observed-extrema assumptions are not used. A nonpositive lower bound is inconclusive.

Reproduce the per-start prediction cache with the main study command (see [results](results.md)); then run `python experiments/pendulum_certificates.py`. No further training is involved.
