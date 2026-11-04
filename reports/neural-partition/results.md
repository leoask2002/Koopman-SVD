# Exploratory two-class calibration on fresh seeds

The original [learned-feature smoke](../neural-transport/results.md) suggested that short neural fits capture some signal but do not reach tabular accuracy. Its exact predictive modes take two values. This follow-up tests whether replacing a fitted rank-two model's nonconstant left and right scores with two-valued features helps. This exploits a property designed into the benchmark; it is not a generic NeuralSVD improvement or a claim of new theory.

Protocol: use entirely new seeds 31, 32, 33 (same generator, sample sizes, network, 64-update budget, validation checkpoint rule and exact conditional-mean metric). Fix both cut rules and all comparison arms before evaluating the fresh tests. The median rule splits each training marginal at its empirical median, with ties going to the lower group. The two-means rule picks an exact one-dimensional least-squares cut via the sorted unique training scores and their counts. Neither uses class labels. Apply each cut separately to both sides of every time step, then refit singular values, overlaps and decoders using training paths only.

| Model and score rule | h1 | h2 | h3 | Mean ± seed SD |
|---|---:|---:|---:|---:|
| full_table_edmd | 0.0832 | 0.0810 | 0.0900 | 0.0847 ± 0.0130 |
| rank2_onehot/soft | 0.0883 | 0.0674 | 0.0590 | 0.0716 ± 0.0128 |
| rank2_onehot/median | 0.1629 | 0.1443 | 0.1507 | 0.1526 ± 0.0851 |
| rank2_onehot/two_means | 0.0240 | 0.0221 | 0.0243 | 0.0235 ± 0.0026 |
| frozen_neural/soft | 0.8834 | 0.6745 | 0.5743 | 0.7107 ± 0.0059 |
| frozen_neural/median | 0.8989 | 0.6751 | 0.5744 | 0.7161 ± 0.0008 |
| frozen_neural/two_means | 0.8978 | 0.6751 | 0.5744 | 0.7158 ± 0.0004 |
| nested_neural/soft | 0.2999 | 0.2658 | 0.2625 | 0.2761 ± 0.0146 |
| nested_neural/median | 0.1500 | 0.1373 | 0.1528 | 0.1467 ± 0.1367 |
| nested_neural/two_means | 0.0240 | 0.0221 | 0.0243 | 0.0235 ± 0.0026 |

With the same nested neural dictionaries, two-means changes mean RMSE from 0.2761 to 0.0235 (-91.5%). The identically partitioned tabular rank-two baseline gives 0.0235. Compare hard neural to hard tabular, not only to frozen networks or raw EDMD.

Limits: median cuts are sensitive to unequal class prevalence and score ties. Two-means also assumes two predictive classes; it can remove necessary within-class variation. For example, with X uniform on [−1,1] and p(y|x)=(1+cxy)/2 on [−1,1] (0<c≤1), the exact mean of Y is cx/3. Even an optimal decoder using only sign(X) predicts c sign(X)/6, leaving an irreducible RMSE c/(6√3) relative to that conditional mean. These three fresh seeds are an exploratory replication on the same favorable generator, not evidence of transfer to new applications.

Reproduce: `PYTHONPATH=src python experiments/neural_transport_partition.py`.
