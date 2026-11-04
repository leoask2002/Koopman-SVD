"""Small learned-vs-frozen NeuralSVD transport check with exact nonlinear truth.

Run from the repository root after installing the package (or with PYTHONPATH=src).
The chosen ranks, splits, three seeds and training budget are fixed below.
"""
from __future__ import annotations

import argparse
import itertools
import json
import platform
from pathlib import Path

import numpy as np
import torch

from koopman_svd.model import NeuralKoopmanSVD
from koopman_svd.transport import TimeIndexedKoopmanSVD

STATES = np.asarray(list(itertools.product((-1.0, 1.0), repeat=5)))
BIT_WEIGHTS = 2 ** np.arange(4, -1, -1)
TRIPLES = ((0, 1, 2), (1, 2, 3), (0, 2, 3), (0, 1, 3))
NUISANCE_P = (0.25, 0.7, 0.4, 0.8)
CORRELATIONS = np.array([0.9, -0.75, 0.85])
SEEDS = (0, 1, 2)
SPLITS = (640, 320, 4096)
TRAIN_STEPS = 64
HIDDEN = (12,)
RANK = 2


def observable(t):
    def evaluate(x):
        return np.sign(np.asarray(x)[:, TRIPLES[t]].sum(axis=1, keepdims=True))
    return evaluate


def state_index(x):
    return ((x > 0).astype(int) @ BIT_WEIGHTS).astype(int)


def onehot(x):
    return np.eye(len(STATES))[state_index(x)]


def distribution_and_kernels():
    rho = np.array([np.where(STATES[:, -1] > 0, p, 1-p) / 16 for p in NUISANCE_P])
    f = [observable(t)(STATES).ravel() for t in range(4)]
    kernels = np.array([rho[t+1][None, :] * (1 + c * np.outer(f[t], f[t+1]))
                        for t, c in enumerate(CORRELATIONS)])
    if np.min(kernels) < 0 or not np.allclose(kernels.sum(axis=-1), 1):
        raise RuntimeError("invalid transition matrix")
    if not np.allclose(np.einsum("ti,tij->tj", rho[:-1], kernels), rho[1:]):
        raise RuntimeError("incorrect marginal propagation")
    # Independent verification of the analytic conditional mean at each horizon.
    power = np.eye(len(STATES))
    for t, kernel in enumerate(kernels):
        power = power @ kernel
        if not np.allclose(power @ f[t+1], np.prod(CORRELATIONS[:t+1]) * f[0]):
            raise RuntimeError("analytic conditional mean disagrees with enumeration")
    return rho, kernels


def paths(rng, n, rho, kernels):
    idx = rng.choice(len(STATES), size=n, p=rho[0])
    result = [STATES[idx]]
    for kernel in kernels:
        cdf = np.cumsum(kernel[idx], axis=1)
        cdf[:, -1] = 1
        idx = (rng.random((n, 1)) > cdf).sum(axis=1)
        result.append(STATES[idx])
    return np.stack(result, axis=1)


def tabular_forecasts(train, x):
    """Full one-hot EDMD / empirical transition matrices (one pseudo-observation per row)."""
    product = np.eye(len(STATES))
    out = []
    for t in range(len(CORRELATIONS)):
        counts = onehot(train[:, t]).T @ onehot(train[:, t + 1])
        marginal = counts.sum(axis=0) / len(train)
        kernel = (counts + marginal) / (counts.sum(axis=1, keepdims=True) + 1)
        product = product @ kernel
        out.append((product @ observable(t+1)(STATES))[state_index(x)].ravel())
    return np.stack(out)


def run(seed):
    rho, kernels = distribution_and_kernels()
    seeds = np.random.SeedSequence(seed).spawn(3)
    train, val, test = [paths(np.random.default_rng(s), n, rho, kernels)
                        for s, n in zip(seeds, SPLITS)]
    x = test[:, 0]
    truth = np.stack([(np.prod(CORRELATIONS[:h]) * observable(0)(x)).ravel()
                      for h in range(1, len(CORRELATIONS)+1)])
    metrics, diagnostics = {}, {}

    def record(name, forecasts):
        if forecasts.shape != truth.shape or not np.isfinite(forecasts).all():
            raise ValueError(f"invalid {name} forecast")
        metrics[name] = np.sqrt(np.mean((forecasts - truth)**2, axis=1)).tolist()

    def evaluate(model):
        return np.stack([model.predict(x, start=0, horizon=h, observable=observable(h)).ravel()
                         for h in range(1, len(CORRELATIONS)+1)])

    record("full_table_edmd", tabular_forecasts(train, x))
    for name, dictionary in [("rank2_coordinates", None), ("rank2_onehot", onehot)]:
        fit = TimeIndexedKoopmanSVD.fit(train, rank=RANK, left_features=dictionary,
                                        right_features=dictionary)
        record(name, evaluate(fit))

    for name, steps in [("frozen_neural", 0), ("nested_neural", TRAIN_STEPS)]:
        models = []
        logs = []
        for t in range(len(CORRELATIONS)):
            model = NeuralKoopmanSVD(input_dim=5, rank=RANK, hidden_dims=HIDDEN,
                                     seed=seed*100 + t, dtype=torch.float64, device="cpu")
            model.fit(train[:, t], train[:, t+1], validation_data=(val[:, t], val[:, t+1]),
                      steps=steps, batch_size=128, lr=0.012, validation_every=8,
                      max_seconds=None, patience=None)
            models.append(model)
            logs.append({key: model.fit_summary_[key] for key in
                         ["steps_completed", "best_step", "best_validation_loss", "trained"]})
        fit = TimeIndexedKoopmanSVD.from_neural_models(train, models, rank=RANK)
        record(name, evaluate(fit))
        diagnostics[name] = logs
    return {"seed": seed, "conditional_mean_rmse": metrics, "checkpoints": diagnostics}


def report(output, rows):
    names = list(rows[0]["conditional_mean_rmse"])
    summary = {}
    for name in names:
        scores = np.array([r["conditional_mean_rmse"][name] for r in rows])
        summary[name] = {"means_by_horizon": scores.mean(axis=0).tolist(),
                         "mean_of_horizon_rmse": float(scores.mean()),
                         "sd_of_seed_mean": float(scores.mean(axis=1).std())}
    output.mkdir(parents=True, exist_ok=True)
    payload = {"protocol": {"seeds": list(SEEDS), "train_val_test": list(SPLITS),
                            "rank_including_constant": RANK, "nested_steps_per_model": TRAIN_STEPS,
                            "hidden": list(HIDDEN), "nuisance_positive_probabilities": list(NUISANCE_P),
                            "lambda": CORRELATIONS.tolist(), "triples_zero_indexed": TRIPLES,
                            "prediction_start": 0, "primary": "equal-weight arithmetic mean of the "
                            "three absolute conditional-mean RMSEs; no trajectory noise added to truth"},
               "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__)},
               "runs": rows, "summary": summary}
    (output / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Neural singular functions across changing dynamics",
        "",
        "This bounded CPU experiment compares actual nested-loss neural optimization against "
        "the same networks at initialization and two tabular benchmarks. It uses a synthetic "
        "system explicitly constructed to have an exact rank-two Koopman representation.",
        "",
        "At time t the observable f_t is the ±1 majority of three observed bits; the triple varies "
        "by time. The fifth bit is a nuisance coordinate whose positive probability changes from "
        "0.25 to 0.70, 0.40 and 0.80. The first four bits remain uniformly distributed at each time. "
        "The transition law is P_t(y|x) = ρ_{t+1}(y)[1 + λ_t f_t(x) f_{t+1}(y)], "
        "with λ=(0.90, −0.75, 0.85). Each row sums to one and maps ρ_t to ρ_{t+1}. "
        "Consequently the exact mean of f_h given X_0=x is the product of the first h λ's "
        "times f_0(x); the script also verifies this by explicit transition-matrix multiplication.",
        "",
        "The primary endpoint is the equal-weight mean of the three horizon-wise conditional-mean "
        "RMSEs, in units of an observable ranging from −1 to 1. Every method sees the same "
        "training paths and test starts. There are 640 training, 320 validation and 4,096 test "
        "trajectories per seed, split before pair extraction. The nonlinear target is supplied "
        "to each decoder; its analytic expectation is never used for fitting or checkpoint selection.",
        "",
        "| Model | h=1 | h=2 | h=3 | Primary mean ± seed SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, s in summary.items():
        lines.append("| " + name + " | " + " | ".join(f"{v:.4f}" for v in s["means_by_horizon"]) +
                     f" | {s['mean_of_horizon_rmse']:.4f} ± {s['sd_of_seed_mean']:.4f} |")
    trained = summary["nested_neural"]["mean_of_horizon_rmse"]
    frozen = summary["frozen_neural"]["mean_of_horizon_rmse"]
    reference = min(("rank2_onehot", "full_table_edmd"), key=lambda n:summary[n]["mean_of_horizon_rmse"])
    lines.extend(["", f"Optimization changes the primary RMSE from {frozen:.4f} (frozen) to "
                  f"{trained:.4f} (nested), a {(trained/frozen-1)*100:+.1f}% change. "
                  f"The stronger of the two tabular controls is {reference} at "
                  f"{summary[reference]['mean_of_horizon_rmse']:.4f}.",
                  "",
                  "The two neural arms use identical rank-two, width-12 tanh MLPs, initialization, "
                  "normalization, post-fit calibration and time-specific overlap matrices. The only "
                  "difference is 64 Adam updates per time step (three steps per seed, 576 updates "
                  "total across three seeds); checkpointing uses validation spectral loss. "
                  "Rank-two coordinate and one-hot arms use the same transport estimator with fixed "
                  "dictionaries. Full-table EDMD counts all 32 states, adding one empirical-marginal "
                  "pseudo-observation per row to handle sparsity. SD quantifies variation across "
                  "three seeds, not a confidence interval. No settings were selected on test errors.",
                  "",
                  "This checks whether representation learning beats its own initialization for "
                  "a known low-rank nonlinear system. The target and operator were chosen to admit "
                  "an exact rank-two model, the finite state table is available, and no "
                  "advantage over generic real-world systems follows. This is not a test of "
                  "certificate coverage, and a forecast is a conditional mean, not a sample path.",
                  "",
                  "Reproduce: `PYTHONPATH=src python experiments/neural_transport_smoke.py` "
                  "(or run directly after installing the project). The JSON contains all per-seed metrics "
                  "and checkpoint selections.", ""])
    (output / "results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/neural-transport"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    report(args.output, [run(seed) for seed in SEEDS])
