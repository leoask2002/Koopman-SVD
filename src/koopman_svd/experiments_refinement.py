"""Bounded comparison of a separate state-augmented EDMD forecasting head.

Synthetic-only model selection uses *known validation trajectories* at
horizons 1, 5, and 10. Exact trajectories starting at test observations are used
only after selecting dictionary rank and ridge. This experiment does not tune
or claim to improve the NeuralSVD decomposition or its singular values.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .experiments import EDMD, cases, exact_future, relative_rmse, serializable
from .forecast import StateAugmentedEDMD
from .model import NeuralKoopmanSVD


HORIZONS = (1, 5, 10)
RIDGES = (0.0, 1e-8, 1e-5, 1e-3)


def select_head(model, x, y, validation_x, validation_truth):
    """Fit all heads on training pairs; select by prespecified validation loss."""
    full = StateAugmentedEDMD(model).fit(x, y)
    best, candidates = None, []
    for rank in range(1, full.supported_rank_ + 1):
        for ridge in RIDGES:
            head = StateAugmentedEDMD(model, rank=rank, ridge=ridge).fit(x, y)
            errors = {
                h: relative_rmse(head.predict(validation_x, steps=h), validation_truth[h])
                for h in HORIZONS
            }
            score = float(np.mean([v * v for v in errors.values()]))
            candidate = {"rank": rank, "ridge": ridge, "score": score,
                         "validation_relative_rmse": errors}
            candidates.append(candidate)
            if best is None or score < best[0]:
                best = (score, head, candidate)
    return best[1], best[2], candidates


def run_refinement(case, seed, steps, max_seconds):
    x, y = case["train"]
    validation_x = case["validation"][0]
    start = time.monotonic()
    model = NeuralKoopmanSVD(case["dim"], case["rank"], hidden_dims=case["hidden"], seed=seed)
    model.fit(x, y, validation_data=case["validation"], steps=steps, batch_size=256,
              lr=0.003, patience=20, max_seconds=max_seconds)
    reference = EDMD(case["degree"]).fit(x, y)
    random_model = NeuralKoopmanSVD(case["dim"], case["rank"], hidden_dims=case["hidden"], seed=seed)
    random_model.fit(x, y, steps=0)
    # These validation rollouts are available because the synthetic maps are
    # known. Real deployment requires held-out measured validation trajectories.
    validation_truth = {h: exact_future(case, validation_x, h) for h in HORIZONS}
    selected, choice, candidates = select_head(model, x, y, validation_x, validation_truth)
    diagnostics = selected.spectral_diagnostics()
    eigenvalues = diagnostics.pop("eigenvalues")
    diagnostics["eigenvalues_real_imag"] = np.column_stack((eigenvalues.real, eigenvalues.imag))
    # All fitting and choices are complete before any test forecasts are read.
    test_x = case["test"][0]
    test_truth = {h: exact_future(case, test_x, h) for h in HORIZONS}
    metrics = {}
    validation_metrics = {}
    methods = {
        "standard_neural_svd": model,
        "state_augmented_edmd": selected,
        "reference_polynomial_edmd": reference,
        "untrained_neural_svd": random_model,
    }
    for name, estimator in methods.items():
        metrics[name] = {}
        validation_metrics[name] = {}
        for h in HORIZONS:
            prediction = estimator.predict(test_x, steps=h)
            truth = test_truth[h]
            rmse = float(np.sqrt(np.mean((prediction - truth) ** 2)))
            metrics[name][h] = {
                "relative_rmse": relative_rmse(prediction, truth),
                "rmse": rmse,
                "error_over_initial_rms": rmse / float(np.sqrt(np.mean(test_x ** 2))),
            }
            validation_metrics[name][h] = relative_rmse(
                estimator.predict(validation_x, steps=h), validation_truth[h])
    return serializable({
        "case": case["name"], "seed": seed, "steps_requested": steps,
        "n_train": len(x), "n_validation": len(validation_x), "n_test": len(test_x),
        "neural_fit": model.fit_summary_, "neural_rank": case["rank"],
        "selected_head": choice, "selection_candidates": candidates,
        "head_spectral_diagnostics": diagnostics,
        "test_metrics": metrics, "validation_relative_rmse": validation_metrics,
        "seconds": time.monotonic() - start,
    })


def write_results(payload, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    lines = [
        "# State-augmented EDMD forecasting refinement", "",
        "This is a separate post-fit forecasting head, not an improvement to the NeuralSVD decomposition or its singular values.",
        "Its dictionary is [1, observed state, calibrated learned left functions]. All scaling and regression use training transitions only.",
        "Dictionary rank and ridge are chosen by the mean squared validation relative RMSE over horizons 1, 5, and 10.",
        "Known synthetic dynamics provide multi-step validation trajectories. Real data would require measured held-out validation trajectories.",
        "Independent test initial states are evaluated only after selection. No test metric selects rank, ridge, or a neural checkpoint.",
        "The benchmark design was developed iteratively; these are exploratory synthetic results, not a confirmatory holdout study.",
        "Relative RMSE is sqrt(mean squared prediction error / mean squared ground truth); absolute and initial-state-normalized errors are also in results.json.", "",
        "| Case | Seed | Head rank | Ridge | Standard 1-step | Augmented 1-step | Polynomial 1-step | Standard 10-step | Augmented 10-step | Polynomial 10-step |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    percent = lambda v: f"{100*v:.3f}%"
    for run in payload["runs"]:
        metric = run["test_metrics"]
        choice = run["selected_head"]
        values = [
            metric[name][str(h)]["relative_rmse"]
            for h in (1, 10)
            for name in ("standard_neural_svd", "state_augmented_edmd", "reference_polynomial_edmd")
        ]
        lines.append(f"| {run['case']} | {run['seed']} | {choice['rank']} | {choice['ridge']:g} | "
                     + " | ".join(map(percent, values)) + " |")
    lines += ["", "## Limits and interpretation", "",
              "- State augmentation changes the forecasting dictionary and its regression objective; it does not repair or re-estimate the neural singular decomposition.",
              "- The polynomial triangular baseline contains its exact closure, so its near-zero error is expected. The unconstrained neural dictionary has no such guarantee.",
              "- Logistic chaos limits long-horizon state prediction: low one-step error does not imply useful individual trajectories at ten steps.",
              "- Large triangular relative errors can result from decaying ground-truth state magnitudes. Inspect the absolute and initial-RMS-normalized errors as well.",
              "- EDMD matrix powers are unconstrained; spectral radius and numerical conditioning are reported separately. No stability guarantee is imposed.",
              "- Three small synthetic seeds establish reproducibility, not general superiority or statistical significance.", "",
              "## Complete horizon comparisons", "",
              "| Case | Seed | Method | 1-step | 5-step | 10-step |", "|---|---:|---|---:|---:|---:|"]
    for run in payload["runs"]:
        for name, metrics in run["test_metrics"].items():
            values = [metrics[str(h)]["relative_rmse"] for h in HORIZONS]
            lines.append(f"| {run['case']} | {run['seed']} | {name} | " + " | ".join(map(percent, values)) + " |")
    lines += ["", "## Reproduction metadata", "", "```json",
              json.dumps(payload["metadata"], indent=2), "```", ""]
    (output / "results.md").write_text("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=Path("reports/forecast"))
    parser.add_argument("--quick", action="store_true", help="100 updates and 1024 training pairs")
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 5000 or not np.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("steps must be 1..5000 and max-seconds must be finite and positive")
    torch.set_num_threads(1)
    steps, n = (100, 1024) if args.quick else (args.steps, 4096)
    payload = {"metadata": {
        "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
        "device": "cpu", "threads": 1, "seeds": args.seeds, "steps": steps, "n_train": n,
        "max_seconds_per_neural_fit": args.max_seconds, "horizons": HORIZONS,
        "ridge_grid": RIDGES, "design_rcond": 1e-10,
        "selection": "mean squared validation relative RMSE at horizons 1, 5, 10",
        "validation_trajectory_source": "known synthetic dynamics; no test-driven selection",
        "command": "python -m koopman_svd.experiments_refinement " + " ".join(sys.argv[1:]),
    }, "runs": []}
    for seed in args.seeds:
        for case in cases(seed, n):
            if case["name"] not in ("triangular", "logistic"):
                continue
            run = run_refinement(case, seed, steps, args.max_seconds)
            payload["runs"].append(run)
            write_results(payload, args.output)
            standard = run["test_metrics"]["standard_neural_svd"]["1"]["relative_rmse"]
            augmented = run["test_metrics"]["state_augmented_edmd"]["1"]["relative_rmse"]
            print(f"{case['name']} seed={seed}: standard={standard:.6f}, augmented={augmented:.6f}", flush=True)
    print(f"Results written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
