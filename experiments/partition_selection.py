"""Can a finite-sample forecast comparison tell when two-level modes help?"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neural_transport_smoke import (CORRELATIONS, NUISANCE_P, STATES, observable, onehot)
from neural_transport_partition import harden
from koopman_svd.certification import certify_improvement
from koopman_svd.transport import TimeIndexedKoopmanSVD

TASKS = ("binary", "four_level")
CALIBRATION_SIZES = (1024, 8192, 65536)
TRIALS = 64
TRAIN_SIZE = 640
MASTER_SEED = 2026110311
FAMILY_DELTA = 0.05
DELTA_CALL = FAMILY_DELTA / (len(TASKS)*len(CALIBRATION_SIZES)*2)


def system(task):
    rho = np.array([np.where(STATES[:, -1] > 0, p, 1-p) / 16 for p in NUISANCE_P])
    if task not in TASKS:
        raise ValueError("unknown task")
    amplitude = np.ones(len(STATES)) if task == "binary" else np.where(STATES[:, -1] > 0, 1., .15)
    scores = np.stack([observable(t)(STATES).ravel()*amplitude for t in range(len(rho))])
    matrices = np.stack([rho[t+1][None, :]*(1 + c*np.outer(scores[t], scores[t+1]))
                         for t, c in enumerate(CORRELATIONS)])
    if np.min(matrices) < 0 or not np.allclose(matrices.sum(axis=2), 1) or not np.allclose(
            np.einsum("ti,tij->tj", rho[:-1], matrices), rho[1:]):
        raise RuntimeError("transition or marginal verification failed")
    reference = []
    product = np.eye(len(STATES))
    scale = 1.
    for h, matrix in enumerate(matrices, 1):
        product = product @ matrix
        reference.append(product @ scores[h])
        scale *= CORRELATIONS[h-1]*np.dot(rho[h], scores[h]**2)
        if not np.allclose(reference[-1], scale*scores[0]):
            raise RuntimeError("independent checks disagree on exact conditional mean")
    return rho, matrices, scores, np.stack(reference)


def simulated_indices(rng, n, rho, matrices):
    indices = np.empty((n, len(matrices)+1), dtype=int)
    indices[:, 0] = rng.choice(len(STATES), n, p=rho[0])
    for t, kernel in enumerate(matrices):
        cdf = np.cumsum(kernel, axis=1)
        cdf[:, -1] = 1.
        u = rng.random(n)
        for i, row in enumerate(cdf):
            mask = indices[:, t] == i
            indices[mask, t+1] = np.searchsorted(row, u[mask], side="right")
    return indices


def evaluate(task, seed):
    rho, matrices, scores, reference = system(task)
    train_rng, calib_rng = map(np.random.default_rng, seed.spawn(2))
    path_ids = simulated_indices(train_rng, TRAIN_SIZE, rho, matrices)
    train = STATES[path_ids]
    soft = TimeIndexedKoopmanSVD.fit(train, rank=2, left_features=onehot, right_features=onehot)
    hard = harden(soft, train, "two_means")
    forecast = np.empty((2, len(matrices), len(STATES)))
    for j, model in enumerate((soft, hard)):
        for h in range(1, len(matrices)+1):
            # Observable table lookup changes by phase; use observed states only.
            fn = lambda x, h=h: onehot(x) @ scores[h]
            forecast[j, h-1] = np.clip(model.predict(STATES, horizon=h, observable=fn).ravel(), -1., 1.)
    risk = np.sum((forecast - reference[None])**2 * rho[0], axis=2)
    gain = risk[0] - risk[1]

    ids = simulated_indices(calib_rng, max(CALIBRATION_SIZES), rho, matrices)
    predictions = forecast[:, :, ids[:, 0]]
    outcomes = np.stack([scores[h, ids[:, h]] for h in range(1, len(matrices)+1)])
    certificates = {}
    for n in CALIBRATION_SIZES:
        args = predictions[:, :, :n], outcomes[:, :n]
        lower = certify_improvement(*args, baseline=0, bounds=(-1, 1), delta=DELTA_CALL).lower[1]
        upper = -certify_improvement(*args, baseline=1, bounds=(-1, 1), delta=DELTA_CALL).lower[0]
        decision = np.where(lower > 0, 1, np.where(upper < 0, -1, 0))
        certificates[str(n)] = {
            "gain_lower": lower.tolist(), "gain_upper": upper.tolist(),
            "decision": decision.tolist(),
            "covers_all_horizons": bool(np.all((lower <= gain) & (gain <= upper))),
            "selected_risk": np.where(decision == 1, risk[1], risk[0]).tolist(),
        }
    return {"individual_risk_mse": risk.tolist(), "true_gain_mse": gain.tolist(),
            "sizes": certificates}


def report(rows, output):
    summary = {}
    for n in CALIBRATION_SIZES:
        simultaneous = [all(trial[task]["sizes"][str(n)]["covers_all_horizons"] for task in TASKS)
                        for trial in rows]
        size_results = {"joint_coverage_at_size": float(np.mean(simultaneous)), "tasks": {}}
        for task in TASKS:
            cells = [trial[task]["sizes"][str(n)] for trial in rows]
            decision = np.array([cell["decision"] for cell in cells])
            risks = np.asarray([trial[task]["individual_risk_mse"] for trial in rows])
            gain = risks[:, 0] - risks[:, 1]
            selected = np.array([cell["selected_risk"] for cell in cells])
            rate = lambda x: float(np.mean(x))
            rmse = lambda x: float(np.sqrt(x).mean())
            size_results["tasks"][task] = {
                "hard_win_rate": rate(decision==1), "soft_win_rate": rate(decision==-1),
                "inconclusive_rate": rate(decision==0),
                "wrong_certified_direction_rate": rate(((decision==1)&(gain<0))|((decision==-1)&(gain>0))),
                "better_model_chosen_rate": rate(np.isclose(selected, np.min(risks, axis=1), rtol=0, atol=1e-14)),
                "soft_rmse": rmse(risks[:, 0]), "hard_rmse": rmse(risks[:, 1]),
                "selected_rmse": rmse(selected), "oracle_rmse": rmse(np.min(risks, axis=1)),
                "inconclusive_rate_by_horizon": np.mean(decision==0, axis=0).tolist(),
            }
        summary[str(n)] = size_results
    covered = sum(all(trial[t]["sizes"][str(n)]["covers_all_horizons"]
                      for t in TASKS for n in CALIBRATION_SIZES) for trial in rows)
    # Exact Clopper-Pearson one-sided lower bound when no failures were observed.
    all_pass_lower = (0.05 ** (1/len(rows))) if covered == len(rows) else None
    payload = {"protocol": {"tasks": list(TASKS), "n_train": TRAIN_SIZE, "sizes": list(CALIBRATION_SIZES),
                            "independent_trials": len(rows), "master_seed": MASTER_SEED,
                            "delta_per_direction_task_size": DELTA_CALL,
                            "family_delta": FAMILY_DELTA,
                            "menu": ["soft tabular rank2", "same scores + train two-means"]},
               "all_sizes_tasks_horizons_covered_trials": covered,
               "summary": summary, "trials": rows}
    output.mkdir(parents=True, exist_ok=True)
    (output/"results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    lines = ["# Calibration chooses when to impose two predictive classes", "",
             "A finite-sample selector compares the **same fitted** time-dependent rank-two tabular "
             "model before and after partitioning its first nonconstant left/right singular scores "
             "with training-only two-means. The rule switches to two-level factors only if a "
             "paired loss-difference lower bound is positive; otherwise it retains soft factors. "
             "A negative upper bound provides evidence that hardening is harmful; a straddling "
             "interval is explicitly inconclusive. No neural networks were trained in this experiment.",
             "", "## Design and exact risk", "",
             "Two 32-state time-varying systems share the setup of the [neural comparison]"
             "(../neural-transport/results.md): changing marginals ρ_t and transitions "
             "P_t(y|x)=ρ_{t+1}(y)[1+λ_t f_t(x) f_{t+1}(y)]. In the binary system f_t is a "
             "majority-of-three sign. In the four-level system multiply the same sign by 1 "
             "when the fifth bit is positive and 0.15 otherwise. In both cases Eρ_t[f_t]=0. "
             "The exact h-step conditional mean of the bounded outcome f_h is f_0 times the "
             "product of λ_t Eρ_{t+1}[f_{t+1}²]; multiplication of enumerated transition "
             "matrices independently verifies the formula. Evaluate each clipped predictor "
             "against this mean over all 32 initial states weighted by ρ_0 (no test Monte Carlo error).",
             "", "Each fresh independent trial fits both candidates on 640 training paths for "
             "each task. Independent calibration starts each supply one stochastic continuation. "
             "Endpoint noise cancels in the population difference of squared losses against "
             "that same outcome. Run the [empirical-Bernstein comparison](../../docs/certification.md) "
             "in both directions. Allocate 0.05/(2 tasks × 2 directions × 3 fixed sizes) per call; "
             "each call simultaneously handles the three horizons. The resulting 95% family "
             "covers all 18 intervals per trial. Calibration sizes are prespecified nested prefixes. "
             "Predeclare all three sizes—no unlimited sampling until significance. Both candidates "
             "are clipped to [−1,1] at deployment, certification and exact-risk evaluation.",
             "", "| Task | Calibration starts | Certified hard | Certified soft | Inconclusive | Selected RMSE | Oracle RMSE |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        for n in CALIBRATION_SIZES:
            e = summary[str(n)]["tasks"][task]
            lines.append(f"| {task} | {n:,} | {e['hard_win_rate']:.1%} | {e['soft_win_rate']:.1%} | "
                         f"{e['inconclusive_rate']:.1%} | {e['selected_rmse']:.4f} | {e['oracle_rmse']:.4f} |")
    lines += ["", "Percentages pool the three horizons across fitted-model trials; these horizons "
              "are not independent. RMSE entries average per-horizon, per-fit square roots of exact "
              "population MSE. Oracle means whichever of the two fitted candidates has lower exact "
              "risk at that horizon; the selector never sees that information. Inconclusive "
              "comparisons default to soft, so an inconclusive interval can coincide with the oracle.",
              "", "| Task | Always soft RMSE | Always hard RMSE |",
              "|---|---:|---:|"]
    for task, e in summary[str(CALIBRATION_SIZES[0])]["tasks"].items():
        lines.append(f"| {task} | {e['soft_rmse']:.4f} | {e['hard_rmse']:.4f} |")
    lines += ["", f"All {len(TASKS)*len(CALIBRATION_SIZES)*3} intervals jointly contained the "
              f"true differences in {covered}/{len(rows)} independent trials."
              + (f" The one-sided 95% binomial lower confidence limit on this generator's "
                 f"joint coverage probability is {all_pass_lower:.3f}." if all_pass_lower is not None else ""),
              "This is a stress test, not a substitute for the concentration argument. "
              "At smaller sizes the bound can be valid and still offer no decision. The experiment "
              "tests binary versus four-level finite modes on one generator family. It does not "
              "establish generic validity on a single dependent rollout, under an arbitrary "
              "deployment shift, for unbounded observables, or for neural models in particular.",
              "", "Reproduce: `OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python experiments/partition_selection.py`. "
              "The JSON records the population risks, confidence intervals and per-horizon decisions "
              "for every independent trial, size and task.", ""]
    (output/"results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/partition-selection"))
    args = parser.parse_args()
    trials = []
    for seed in np.random.SeedSequence(MASTER_SEED).spawn(TRIALS):
        trials.append({task: evaluate(task, child) for task, child in zip(TASKS, seed.spawn(len(TASKS)))})
    report(trials, args.output)
