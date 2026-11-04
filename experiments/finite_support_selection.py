"""Matched global-vs-known-support paired certificates, on fresh fits and draws."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neural_transport_partition import harden
from neural_transport_smoke import STATES, onehot
from partition_selection import TASKS, TRAIN_SIZE, simulated_indices, system
from koopman_svd.certification import certify_improvement
from koopman_svd.transport import TimeIndexedKoopmanSVD

METHODS = ("global", "complete_support")
SIZES = (1024, 8192)
TRIALS = 64
SEED = 2026110312  # Disjoint from the earlier selector study.
DELTA = .05/(len(METHODS)*len(TASKS)*len(SIZES)*2)  # both directions; call includes all H


def run_task(task, seed):
    rho, matrices, scores, truth = system(task)
    fit_rng, calib_rng = map(np.random.default_rng, seed.spawn(2))
    train = STATES[simulated_indices(fit_rng, TRAIN_SIZE, rho, matrices)]
    soft = TimeIndexedKoopmanSVD.fit(train, rank=2, left_features=onehot, right_features=onehot)
    hard = harden(soft, train, "two_means")
    forecasts = np.stack([np.stack([model.predict(STATES, horizon=h,
                               observable=lambda x, h=h: onehot(x) @ scores[h]).ravel()
                               for h in range(1, len(matrices)+1)]) for model in (soft, hard)])
    # The table is evaluated over every possible start, before drawing calibration.
    table = np.clip(forecasts, -1., 1.)
    risk = np.sum((table - truth[None])**2 * rho[0], axis=-1)
    endpoint = [((table[0]-y)**2 - (table[1]-y)**2) for y in (-1., 1.)]
    width_frac = ((np.maximum(*endpoint).max(axis=-1) -
                   np.minimum(*endpoint).min(axis=-1)) / 8.)
    ids = simulated_indices(calib_rng, max(SIZES), rho, matrices)
    y = np.stack([scores[h, ids[:, h]] for h in range(1, len(matrices)+1)])
    pred = table[:, :, ids[:, 0]]
    intervals = {}
    for n in SIZES:
        intervals[str(n)] = {}
        for method in METHODS:
            kwargs = ({} if method == "global" else
                      dict(support_predictions=table, support_indices=ids[:n, 0]))
            args = pred[:, :, :n], y[:, :n]
            lower = certify_improvement(*args, baseline=0, bounds=(-1,1), delta=DELTA, **kwargs).lower[1]
            upper = -certify_improvement(*args, baseline=1, bounds=(-1,1), delta=DELTA, **kwargs).lower[0]
            intervals[str(n)][method] = np.stack([lower, upper]).tolist()
    return {"risk_soft_hard": risk.tolist(), "diameter_fraction_of_global": width_frac.tolist(),
            "intervals_lower_upper": intervals}


def summarize(rows):
    result = {}
    for task in TASKS:
        truth = np.array([row[task]["risk_soft_hard"] for row in rows])
        gain = truth[:, 0] - truth[:, 1]
        entry = {"always_soft_rmse": float(np.sqrt(truth[:,0]).mean()),
                 "always_hard_rmse": float(np.sqrt(truth[:,1]).mean()),
                 "oracle_rmse": float(np.sqrt(truth.min(axis=1)).mean()),
                 "mean_diameter_fraction_of_global": float(np.mean(
                     [row[task]["diameter_fraction_of_global"] for row in rows])),
                 "sizes": {}}
        for n in SIZES:
            methods = {}
            for method in METHODS:
                bounds = np.array([row[task]["intervals_lower_upper"][str(n)][method] for row in rows])
                accept_hard = bounds[:, 0] > 0
                accept_soft = bounds[:, 1] < 0
                decision = np.where(accept_hard, truth[:, 1], truth[:, 0])
                methods[method] = {
                    "certified_hard": float(accept_hard.mean()),
                    "certified_soft": float(accept_soft.mean()),
                    "inconclusive": float((~accept_soft & ~accept_hard).mean()),
                    "wrong_direction": float(((accept_hard & (gain<0))|(accept_soft & (gain>0))).mean()),
                    "selected_rmse": float(np.sqrt(decision).mean()),
                    "mean_interval_width": float((bounds[:, 1]-bounds[:, 0]).mean()),
                }
            entry["sizes"][str(n)] = methods
        result[task] = entry
    return result


def covered(trial):
    for task in TASKS:
        soft, hard = np.asarray(trial[task]["risk_soft_hard"])
        for n in SIZES:
            for method in METHODS:
                low, high = np.asarray(trial[task]["intervals_lower_upper"][str(n)][method])
                if not np.all((low <= soft-hard) & (soft-hard <= high)):
                    return False
    return True


def write(output, rows):
    summary = summarize(rows)
    covers = sum(covered(row) for row in rows)
    output.mkdir(parents=True, exist_ok=True)
    payload = {"master_seed": SEED, "independent_trials": TRIALS, "calibration_sizes": list(SIZES),
               "family_delta": .05, "delta_per_call": DELTA, "joint_covered_trials": covers,
               "n_train": TRAIN_SIZE, "methods": list(METHODS), "summary": summary}
    (output/"results.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    with (output/"trials.jsonl").open("w") as fp:
        for i, row in enumerate(rows):
            fp.write(json.dumps({"trial":i, **row}, separators=(",",":"), allow_nan=False)+"\n")
    lines = ["# Reducing the calibration budget with a known finite support", "",
        "This experiment compares two confidence bounds for the same pair of fitted tabular "
        "rank-two candidates (continuous-valued singular scores versus training-only two-means) "
        "under the [binary and four-level systems](../partition-selection/results.md). Keep the "
        "soft model unless a positive lower confidence bound establishes that hardening improves "
        "conditional-mean MSE. Report evidence for soft separately from inconclusive decisions.",
        "", "The global comparison uses the full possible range of a paired squared-loss difference. "
        "The refined comparison uses its exact range over all **32 possible initial states** and "
        "the two observable endpoints −1 and +1. The table is obtained after fitting and before "
        "calibration; no empirical extrema, true transition probabilities or conditional means "
        "are used in constructing the bound. Both use the same empirical variance, calibration "
        "draws, family budget, prediction clipping and two-sided gain definition. See the "
        "[proof and API](../../docs/certification.md).",
        "", f"Fresh master seed {SEED}; {TRIALS} independent refits, 640 training paths per task, "
        "and nested calibration sizes fixed at 1,024 and 8,192. Each start supplies one independent "
        "trajectory (three correlated horizons). Allocate 0.05/(2 methods × 2 tasks × 2 sizes × 2 "
        "directions) per call; each call covers all horizons. Hence all 24 two-sided intervals "
        "per trial have ≥95% simultaneous confidence. Exact risk sums over the entire initial law "
        "using conditional means independently verified by transition-matrix multiplication.",
        "", "| Task | Starts | Score range | Certified hard | Certified soft | Inconclusive | Selected RMSE | Oracle RMSE |",
        "|---|---:|---|---:|---:|---:|---:|---:|"]
    for task, t in summary.items():
        for n in SIZES:
            for method, e in t["sizes"][str(n)].items():
                lines.append(f"| {task} | {n:,} | {method} | {e['certified_hard']:.1%} | "
                             f"{e['certified_soft']:.1%} | {e['inconclusive']:.1%} | "
                             f"{e['selected_rmse']:.4f} | {t['oracle_rmse']:.4f} |")
    lines += ["", "Decision percentages pool three horizons within each fit; they are not "
              "independent sampling units. RMSE is averaged over horizons and fitted-model trials; "
              "the oracle chooses the lower true-risk candidate separately for each fit and horizon.",
              "", "| Task | Always soft RMSE | Always hard RMSE | Mean support/global score diameter |",
              "|---|---:|---:|---:|"]
    for task, e in summary.items():
        lines.append(f"| {task} | {e['always_soft_rmse']:.4f} | {e['always_hard_rmse']:.4f} | "
                     f"{e['mean_diameter_fraction_of_global']:.1%} |")
    lines += ["", f"Joint intervals covered all exact differences in {covers}/{TRIALS} trials."
              + (f" One-sided 95% exact binomial lower limit: {(.05**(1/TRIALS)):.3f}."
                 if covers == TRIALS else ""),
              "This is finite Monte Carlo evidence for the two generators, not proof of uniform "
              "real-world coverage. Inspect the JSON for wrong-direction rates and interval widths. "
              "The tighter range only reduces the range-dependent term in the empirical-Bernstein "
              "radius; the sampling-variance term remains. Small data and small true differences can "
              "still leave most decisions unresolved. A smaller but invalid range taken from sampled "
              "extrema could appear to improve decisions while destroying coverage.",
              "", "Scope: this experiment takes advantage of an exhaustive finite state alphabet. "
              "The results do not transfer automatically to continuous state spaces, uncontrolled "
              "distribution shifts, unknown support, or neural architectures. No neural weights were "
              "trained. Reproduce: `OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python "
              "experiments/finite_support_selection.py`. The JSONL contains risks and confidence "
              "endpoints for every fit, task, size and method.", ""]
    (output/"results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/finite-support-selection"))
    args = parser.parse_args()
    rows = []
    for seed in np.random.SeedSequence(SEED).spawn(TRIALS):
        rows.append({task:run_task(task, child) for task, child in zip(TASKS, seed.spawn(len(TASKS)))})
    write(args.output, rows)
