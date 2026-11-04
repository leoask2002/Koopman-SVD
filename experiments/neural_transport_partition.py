"""Fresh-seed exploration of an explicit two-class inductive bias; see results.md."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch

from neural_transport_smoke import (CORRELATIONS, HIDDEN, RANK, SPLITS, TRAIN_STEPS,
                                    distribution_and_kernels, observable, onehot, paths,
                                    tabular_forecasts)
from koopman_svd.model import NeuralKoopmanSVD
from koopman_svd.transport import TimeIndexedKoopmanSVD

FRESH_SEEDS = (31, 32, 33)


def thresholded(score_fn, data, rule):
    """Return fixed scalar ±1 features using training scores only; no true class labels."""
    values = score_fn(data)[:, 1]
    if rule == "median":
        cut = np.median(values)
    elif rule == "two_means":
        # Exact 1-D squared-error optimum across nonempty intervals.
        unique, count = np.unique(values, return_counts=True)
        if len(unique) <= 1:
            cut = values[0]
        else:
            count_l = np.cumsum(count)[:-1]
            count_r = len(values) - count_l
            sum_l = np.cumsum(unique * count)[:-1]
            sum_r = np.dot(unique, count) - sum_l
            between = count_l * count_r * (sum_l/count_l - sum_r/count_r)**2
            k = int(between.argmax())
            cut = unique[k] + (unique[k+1] - unique[k]) / 2
    else:
        raise ValueError("unknown cut rule")

    def features(x):
        return np.where(score_fn(x)[:, 1:2] > cut, 1., -1.)
    return features


def harden(model, train, rule):
    left, right = [], []
    for t, step in enumerate(model.steps):
        left.append(thresholded(step.left_functions, train[:, t], rule))
        right.append(thresholded(step.right_functions, train[:, t+1], rule))
    return TimeIndexedKoopmanSVD.fit(train, rank=RANK, left_features=left, right_features=right)


def run(seed):
    rho, kernels = distribution_and_kernels()
    groups = np.random.SeedSequence(seed).spawn(3)
    train, val, test = [paths(np.random.default_rng(s), n, rho, kernels)
                        for s, n in zip(groups, SPLITS)]
    x = test[:, 0]
    truth = np.stack([(np.prod(CORRELATIONS[:h]) * observable(0)(x)).ravel()
                      for h in range(1, len(CORRELATIONS)+1)])

    def measure(forecasts):
        if forecasts.shape != truth.shape or not np.isfinite(forecasts).all():
            raise ValueError("invalid forecast")
        return np.sqrt(np.mean((forecasts - truth)**2, axis=1)).tolist()

    scores = {"full_table_edmd": measure(tabular_forecasts(train, x))}
    models = {"rank2_onehot": TimeIndexedKoopmanSVD.fit(train, rank=RANK, left_features=onehot,
                                                        right_features=onehot)}
    for method, steps in [("frozen_neural", 0), ("nested_neural", TRAIN_STEPS)]:
        nets = []
        for t in range(len(CORRELATIONS)):
            model = NeuralKoopmanSVD(input_dim=5, rank=RANK, hidden_dims=HIDDEN,
                                     seed=100*seed+t, dtype=torch.float64, device="cpu")
            nets.append(model.fit(train[:, t], train[:, t+1],
                                  validation_data=(val[:, t], val[:, t+1]), steps=steps,
                                  batch_size=128, lr=0.012, max_seconds=None,
                                  validation_every=8, patience=None))
        models[method] = TimeIndexedKoopmanSVD.from_neural_models(train, nets, rank=RANK)
    for method, model in models.items():
        for rule in ("soft", "median", "two_means"):
            fitted = model if rule == "soft" else harden(model, train, rule)
            forecasts = np.stack([fitted.predict(x, horizon=h, observable=observable(h)).ravel()
                                  for h in range(1, len(CORRELATIONS)+1)])
            scores[f"{method}/{rule}"] = measure(forecasts)
    return {"seed": seed, "rmse_by_horizon": scores}


def report(output, rows):
    summary = {}
    for name in rows[0]["rmse_by_horizon"]:
        a = np.array([row["rmse_by_horizon"][name] for row in rows])
        summary[name] = {"horizon_means": a.mean(axis=0).tolist(), "mean": float(a.mean()),
                         "seed_sd": float(a.mean(axis=1).std())}
    output.mkdir(parents=True, exist_ok=True)
    payload = {"seeds": list(FRESH_SEEDS), "steps_per_net": TRAIN_STEPS,
               "primary": "mean conditional-mean RMSE equally weighted across the three horizons",
               "versions": {"python": platform.python_version(), "numpy": np.__version__,
                            "torch": str(torch.__version__)}, "runs": rows, "summary": summary}
    (output / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    lines = ["# Exploratory two-class calibration on fresh seeds", "",
             "The original [learned-feature smoke](../neural-transport/results.md) suggested that "
             "short neural fits capture some signal but do not reach tabular accuracy. Its exact "
             "predictive modes take two values. This follow-up tests whether replacing a fitted "
             "rank-two model's nonconstant left and right scores with two-valued features helps. "
             "This exploits a property designed into the benchmark; it is not a generic "
             "NeuralSVD improvement or a claim of new theory.", "",
             "Protocol: use entirely new seeds 31, 32, 33 (same generator, sample sizes, network, "
             "64-update budget, validation checkpoint rule and exact conditional-mean metric). "
             "Fix both cut rules and all comparison arms before evaluating the fresh tests. "
             "The median rule splits each training marginal at its empirical median, with ties "
             "going to the lower group. The two-means rule picks an exact one-dimensional "
             "least-squares cut via the sorted unique training scores and their counts. Neither "
             "uses class labels. Apply each cut separately to both sides of every time step, "
             "then refit singular values, overlaps and decoders using training paths only.", "",
             "| Model and score rule | h1 | h2 | h3 | Mean ± seed SD |",
             "|---|---:|---:|---:|---:|"]
    for name, result in summary.items():
        lines.append("| " + name + " | " + " | ".join(f"{v:.4f}" for v in result["horizon_means"])
                     + f" | {result['mean']:.4f} ± {result['seed_sd']:.4f} |")
    soft = summary["nested_neural/soft"]["mean"]
    hard = summary["nested_neural/two_means"]["mean"]
    tab = summary["rank2_onehot/two_means"]["mean"]
    lines.extend(["", f"With the same nested neural dictionaries, two-means changes mean RMSE "
                  f"from {soft:.4f} to {hard:.4f} ({(hard/soft - 1)*100:+.1f}%). "
                  f"The identically partitioned tabular rank-two baseline gives {tab:.4f}. "
                  "Compare hard neural to hard tabular, not only to frozen networks or raw EDMD.",
                  "",
                  "Limits: median cuts are sensitive to unequal class prevalence and score ties. "
                  "Two-means also assumes two predictive classes; it can remove necessary "
                  "within-class variation. For example, with X uniform on [−1,1] and "
                  "p(y|x)=(1+cxy)/2 on [−1,1] (0<c≤1), the exact mean of Y is cx/3. "
                  "Even an optimal decoder using only sign(X) predicts c sign(X)/6, leaving "
                  "an irreducible RMSE c/(6√3) relative to that conditional mean. These three "
                  "fresh seeds are an exploratory replication on the same favorable generator, "
                  "not evidence of transfer to new applications.",
                  "", "Reproduce: `PYTHONPATH=src python experiments/neural_transport_partition.py`.", ""])
    (output / "results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/neural-partition"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    report(args.output, [run(seed) for seed in FRESH_SEEDS])
