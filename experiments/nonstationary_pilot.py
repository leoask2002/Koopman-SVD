"""Small synthetic identification study; model/rank selection uses calibration only.

Run from repository root: OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python experiments/nonstationary_pilot.py
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from koopman_svd.certification import certify_forecasts, certify_improvement, select_horizon
from koopman_svd.transport import TimeIndexedKoopmanSVD

H = 5
REGIMES = ("stationary", "transient", "switching")


class Chain:
    """6-state chain; circulant kernels preserve uniform mass in the stationary control."""
    name = "finite chain"
    ranks = [1, 3, 5, 6]

    def __init__(self, regime):
        self.mass = np.ones(6) / 6 if regime == "stationary" else np.array([.48, .20, .12, .08, .07, .05])
        directions = [1, -1, 2, -2, 1] if regime == "switching" else [1] * H
        eye = np.eye(6)
        self.P = np.stack([.30 * eye + .58 * np.roll(eye, d, axis=1) + .12 / 6 for d in directions])
        angle = 2 * np.pi * np.arange(6) / 6
        self.y = .65 * np.cos(angle) + .35 * np.sin(2 * angle)

    def starts(self, rng, n):
        return np.eye(6)[rng.choice(6, size=n, p=self.mass)]

    def paths(self, rng, x):
        out = [x]
        for p in self.P:
            cdf = np.cumsum(p[out[-1].argmax(axis=1)], axis=1)
            ids = (rng.uniform(size=(len(x), 1)) > cdf[:, :-1]).sum(axis=1)
            out.append(np.eye(6)[ids])
        return np.stack(out, axis=1)

    def target(self, x):
        return (x @ self.y)[:, None]

    def truth(self, x):
        op = np.eye(6)
        cols = []
        for p in self.P:
            op = op @ p
            cols.append(x @ op @ self.y)
        return np.stack(cols, axis=1)

    def reference(self):
        return np.eye(6), self.mass

    @staticmethod
    def closure(x):
        return x


class WarpedCircle:
    """Smooth noisy rotations, observed through an invertible nonlinear coordinate change."""
    name = "warped circle"
    ranks = [1, 3, 5, 8]

    def __init__(self, regime):
        self.kappa = 0. if regime == "stationary" else 2.
        self.phase = np.array([.5, -.85, 1.05, -.70, .8]) if regime == "switching" else np.full(H, .5)
        self.sd = np.full(H, .18)
        self.keep = .88

    @staticmethod
    def observe(theta):
        w = theta + .45 * np.sin(theta)
        return np.stack([np.cos(w), np.sin(w)], axis=1)

    @staticmethod
    def angle(x):
        w = np.arctan2(x[:, 1], x[:, 0])
        a = w.copy()
        for _ in range(7):
            a -= (a + .45 * np.sin(a) - w) / (1 + .45 * np.cos(a))
        return a

    def starts(self, rng, n):
        return self.observe(rng.vonmises(.85, self.kappa, size=n))

    def paths(self, rng, x):
        a = self.angle(x)
        out = [x]
        for phase, sd in zip(self.phase, self.sd):
            a = np.where(rng.random(len(x)) < self.keep,
                         a + phase + rng.normal(0, sd, len(x)), rng.uniform(-np.pi, np.pi, len(x)))
            out.append(self.observe(a))
        return np.stack(out, axis=1)

    def target(self, x):
        a = self.angle(x)
        return (.65 * np.cos(a) + .35 * np.sin(3 * a))[:, None]

    def truth(self, x):
        a = self.angle(x)[:, None] + np.cumsum(self.phase)
        v = np.cumsum(self.sd ** 2)
        return self.keep ** np.arange(1, H + 1) * (
            .65 * np.cos(a) * np.exp(-v / 2) + .35 * np.sin(3 * a) * np.exp(-9 * v / 2))

    def reference(self, size=1024):
        a = 2 * np.pi * np.arange(size) / size - np.pi
        weights = np.exp(self.kappa * np.cos(a - .85))
        return self.observe(a), weights / weights.sum()

    def closure(self, x):
        a = self.angle(x)
        return np.stack([np.cos(a), np.sin(a), np.cos(3 * a), np.sin(3 * a)], axis=1)


def lift(feature, x):
    return np.column_stack([np.ones(len(x)), feature(x)])


def regression_baselines(paths, feat, observable):
    """One-step projected EDMD (time-specific or pooled) and horizon-specific direct OLS."""
    n, nt, d = paths.shape
    phi = lift(feat, paths.reshape(n * nt, d)).reshape(n, nt, -1)
    y = observable(paths.reshape(n * nt, d)).reshape(n, nt)
    ls = lambda a, b: np.linalg.lstsq(a, b, rcond=1e-10)[0]
    pooled = ls(phi[:, :-1].reshape(-1, phi.shape[2]), phi[:, 1:].reshape(-1, phi.shape[2]))
    c_pool = ls(phi.reshape(-1, phi.shape[2]), y.reshape(-1))
    A = np.eye(phi.shape[2])
    Ac = np.eye(phi.shape[2])
    local, fixed, direct = [], [], []
    for h in range(1, nt):
        A = A @ ls(phi[:, h-1], phi[:, h])
        Ac = Ac @ pooled
        local.append(A @ ls(phi[:, h], y[:, h]))
        fixed.append(Ac @ c_pool)
        direct.append(ls(phi[:, 0], y[:, h]))
    def predictor(coefs):
        mat = np.stack(coefs, axis=1)
        return lambda x: np.clip(lift(feat, x) @ mat, -1, 1)
    return {k: predictor(v) for k, v in (("time EDMD", local), ("pooled EDMD", fixed), ("direct OLS", direct))}


def system_seed(bench, regime, seed):
    # Shared streams within a case for every competing estimator.
    return np.random.SeedSequence([73041, bench, REGIMES.index(regime), seed])


def targets(sys, paths):
    return sys.target(paths[:, 1:].reshape(-1, paths.shape[2])).reshape(len(paths), H).T


def validation_sample(sys, rng, n):
    x = sys.starts(rng, n)
    return x, targets(sys, sys.paths(rng, x)), targets(sys, sys.paths(rng, x))


def run_case(bench, regime, seed, args):
    sys = (Chain, WarpedCircle)[bench](regime)
    rngs = [np.random.default_rng(s) for s in system_seed(bench, regime, seed).spawn(6)]
    train, cal, test, features, repeat, check = rngs
    paths = sys.paths(train, sys.starts(train, args.train))
    inputs = paths.shape[-1]
    W = features.normal(0, 1.3 / np.sqrt(inputs), (inputs, 16))
    bias = features.uniform(-1, 1, 16)
    feat = lambda x: np.tanh(x @ W + bias)
    reference, mass = sys.reference()
    x_test = sys.starts(test, args.test)
    x_cal, y1, y2 = validation_sample(sys, cal, args.cal)
    x_all = np.concatenate([reference, x_test, x_cal])
    n0, n1 = len(reference), len(reference) + len(x_test)
    fit = TimeIndexedKoopmanSVD.fit(paths, rank=max(sys.ranks), left_features=feat, right_features=feat)
    models = regression_baselines(paths, feat, sys.target)
    models["closure EDMD"] = regression_baselines(paths, sys.closure, sys.target)["time EDMD"]

    def compose_predictor(rank, mode):
        model = fit.truncate(rank)
        if mode == "identity":
            model = replace(model, bridges=tuple(np.eye(*b.shape) for b in model.bridges))
        def predict(x):
            return np.clip(np.column_stack([model.predict(
                x, start=0, horizon=h, observable=sys.target
            ).reshape(-1) for h in range(1, H + 1)]), -1, 1)
        return predict

    names = [f"rank {r}" for r in sys.ranks]
    for r, key in zip(sys.ranks, names):
        models[key] = compose_predictor(r, "empirical")
    models["rank 3 no bridge"] = compose_predictor(3, "identity")
    pred = {k: v(x_all) for k, v in models.items()}
    f_ref, f_test = sys.truth(reference), sys.truth(x_test)
    ref_mse = {k: mass @ ((v[:n0] - f_ref) ** 2) for k, v in pred.items()}
    test_rmse = {k: np.mean((v[n0:n1] - f_test)**2, axis=0)**.5 for k, v in pred.items()}
    family = np.stack([pred[k][n1:].T for k in names])
    cert = certify_forecasts(family, y1, y2, bounds=(-1., 1.), delta=args.delta / 2)
    selected = select_horizon(cert, ranks=sys.ranks, tolerance=args.tolerance)
    covered = bool(np.all(np.stack([ref_mse[k] for k in names]) <= cert.upper + 1e-12))
    # Published RMSE for the selected predictor even on horizons outside its certificate,
    # explicitly distinct from its reported contiguous usable horizon.
    if selected.horizon:
        sel = f"rank {selected.rank}"
        test_rmse["selected"] = test_rmse[sel]
        ref_mse["selected"] = ref_mse[sel]
    with_baseline = np.concatenate([family, pred["time EDMD"][n1:].T[None]])
    contrast = certify_improvement(with_baseline, y1, baseline=len(family),
                                   bounds=(-1., 1.), delta=args.delta / 2)
    # Two independent seeds for training versus auxiliary validation of analytic target.
    anchor = sys.starts(check, 1)
    sampled = sys.paths(check, np.repeat(anchor, 20000, axis=0))
    t = targets(sys, sampled)
    check_error = abs(t.mean(axis=1) - sys.truth(anchor).ravel())
    check_se = t.std(axis=1) / np.sqrt(t.shape[1])
    result = {
        "system": sys.name, "regime": regime, "seed": seed, "ranks": sys.ranks,
        "population_rmse_by_horizon": {k: v**.5 for k, v in ref_mse.items()},
        "independent_test_rmse_by_horizon": test_rmse,
        "risk_hat": cert.estimate, "risk_upper": cert.upper,
        "all_ranks_horizons_covered": covered,
        "selected_rank": None if not selected.horizon else selected.rank,
        "usable_horizon": selected.horizon,
        "baseline_gain_lower": contrast.lower[:-1],
        "true_baseline_gain": np.stack([ref_mse["time EDMD"] - ref_mse[k] for k in names]),
        "analytic_mean_mc_check": {"absolute_discrepancy": check_error, "mc_standard_error": check_se},
    }
    if bench == 1:
        grid2, mass2 = sys.reference(2048)
        result["quad_max_mse_discrepancy"] = max(
            np.max(np.abs(ref_mse[k] - mass2 @ ((fn(grid2)-sys.truth(grid2))**2))) for k, fn in models.items())
    if bench == 0 and regime == "switching" and seed == 0:
        numeric_ref = np.stack([ref_mse[k] for k in names])
        covers = 0
        horizon_counts = [0] * (H+1)
        for _ in range(args.replicates):
            xc, o1, o2 = validation_sample(sys, repeat, args.cal)
            pcs = np.stack([pred[k][:n0][xc.argmax(axis=1)].T for k in names])
            rep_cert = certify_forecasts(pcs, o1, o2, bounds=(-1., 1.), delta=args.delta / 2)
            covers += int(np.all(numeric_ref <= rep_cert.upper + 1e-12))
            horizon_counts[select_horizon(rep_cert, ranks=sys.ranks, tolerance=args.tolerance).horizon] += 1
        result["conditional_calibration_repeats"] = {
            "replicates": args.replicates, "simultaneously_covered": covers,
            "usable_horizon_counts": horizon_counts, "fixed_training_seed": seed,
        }
    return result


def report(results, args):
    keys = ["pooled EDMD", "time EDMD", "direct OLS", "closure EDMD", "rank 3", "rank 3 no bridge"]
    lines = [
        "# Forecasting with time-indexed singular spaces: CPU pilot", "",
        "All scores below are absolute conditional-mean RMSE of forecasts clipped to [−1,1]. "
        "Entries average over five horizons first, then paired training seeds. Lower is better. "
        "Analytic conditional means are used only for independent evaluation, never for fitting or selection.",
        "", "| System | Regime | Pooled EDMD | Time EDMD | Direct OLS | Known-closure EDMD | Rank 3 + bridge | Rank 3, identity |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    groups = [(sys, reg, [r for r in results if r["system"] == sys and r["regime"] == reg])
              for sys in ("finite chain", "warped circle") for reg in REGIMES]
    for sys, reg, group in groups:
        means = [np.mean([row["population_rmse_by_horizon"][key] for row in group]) for key in keys]
        lines.append(f"| {sys} | {reg} | " + " | ".join(f"{m:.4f}" for m in means) + " |")
    lines.extend([
        "", "The rank-3 and identity columns have identical data, dictionary, singular functions and readout. "
        "Replacing each cross-basis overlap by an identity deletes basis orientation information; this is "
        "a deliberately incorrect ablation, not a competitive baseline. The pooled, time EDMD and direct "
        "OLS baselines use the same random tanh dictionary with no imposed rank-3 cap. Direct OLS fits a "
        "separate regression for each horizon. Known-closure EDMD has privileged features (tabular or four "
        "latent harmonics) but estimates its dynamics from the same paths.",
        "", "## Selection and certification", "",
        f"The predeclared menu uses ranks {Chain.ranks} on the chain and {WarpedCircle.ranks} on the circle. "
        f"The RMSE tolerance is {args.tolerance:.2f}. The longest acceptable contiguous prefix is selected; "
        "ties prefer smaller rank. Horizon 0 abstains. Bounds cover each complete fitted forecast, "
        "not separate uncertainty components for its factors. At each seed/regime, split the failure budget "
        f"{args.delta:g} equally between the absolute-risk and paired-improvement families. "
        "No single joint guarantee over different seeds or systems is claimed.",
        "", "| System | Regime | Selected rank by seed (0 = abstain) | Certified horizon by seed | Risk family covered | Positive lower bounds vs time EDMD |",
        "|---|---|---|---|---:|---:|",
    ])
    for sys, reg, group in groups:
        ranks = [r["selected_rank"] or 0 for r in group]
        horizons = [r["usable_horizon"] for r in group]
        pos = sum(int((r["baseline_gain_lower"] > 0).sum()) for r in group)
        total = sum(r["baseline_gain_lower"].size for r in group)
        covers = sum(r["all_ranks_horizons_covered"] for r in group)
        lines.append(f"| {sys} | {reg} | {ranks} | {horizons} | {covers}/{len(group)} | {pos}/{total} |")
    reps = next((r["conditional_calibration_repeats"] for r in results if "conditional_calibration_repeats" in r), None)
    if reps:
        lines.extend(["", f"With the switching chain fit fixed at training seed 0, {reps['simultaneously_covered']}/"
                           f"{reps['replicates']} fresh calibration experiments covered all ranks/horizons. "
                           "This is a check at one fixed training realization, not a general coverage estimate."])
    lines.extend([
        "", "## Design and interpretation", "",
        f"- For each of {args.seeds} seeds: {args.train} independent training paths, {args.cal} independent "
        f"calibration starts with two independent continuations, and {args.test} independent test starts. "
        "The public JSON records every model/horizon, signed calibration estimates, MSE upper bounds, "
        "and lower bounds for baseline-minus-candidate MSE. The sample unit is a trajectory start.",
        "- Primary metric fixed before viewing results: mean over horizons 1–5 of population RMSE; primary "
        "spectral model rank 3; primary competitive comparison is time-specific EDMD. "
        "All outputs use one bounded scalar observable. No controller, trajectory-tail or pointwise guarantee is implied.",
        "- Finite chain: six states; each step holds with probability .30, shifts cyclically with probability "
        ".58, and draws uniform with probability .12. Exact risk sums over the starting distribution. "
        "The observable is .65 cos(theta) + .35 sin(2 theta).",
        "- Circle: latent angle advances by a time-specific phase plus Gaussian noise (SD .18) with probability "
        ".88, or resets uniformly. The observed circle angle is theta + .45 sin(theta); the observable is "
        ".65 cos(theta) + .35 sin(3 theta). For harmonic k, the h-step multiplier is "
        ".88^h exp(−k² h .18²/2) exp(i k sum(phase)). Population MSE uses 1,024-point periodic "
        "quadrature. A second 2,048-point grid checks numerical error. A separate single-start Monte Carlo "
        "check compares each analytic horizon against 20,000 simulated continuations (per seed and case).",
        "- Both stationary controls use a uniform starting law and fixed transitions. Transient uses a biased "
        "starting law with the identical homogeneous transition. Switching changes the transitions as well. "
        "Absolute risk is checked against the exact chain sum or the refined numerical circle integral.",
        "- The estimator uses 16 fixed random tanh features per side; it learns the empirical linear algebra "
        "of those features from training paths. This study does not optimize neural weights or establish a "
        "neural advantage. Time indices and synchronized replicate starts are known.",
        "", "## Reproduction", "",
        "```bash",
        f"OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python experiments/nonstationary_pilot.py "
        f"--seeds {args.seeds} --train {args.train} --cal {args.cal} --test {args.test} "
        f"--replicates {args.replicates} --tolerance {args.tolerance:g} --delta {args.delta:g}",
        "```", "",
    ])
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key, default in (("seeds", 5), ("train", 1024), ("cal", 4096), ("test", 2048), ("replicates", 64)):
        p.add_argument(f"--{key}", type=int, default=default)
    p.add_argument("--tolerance", type=float, default=.30)
    p.add_argument("--delta", type=float, default=.05)
    p.add_argument("--out", type=Path, default=Path("reports/nonstationary"))
    args = p.parse_args()
    if min(args.seeds, args.train, args.test, args.replicates) < 1 or args.cal < 2:
        p.error("positive budgets and at least two calibration starts are required")
    results = []
    for b in range(2):
        for reg in REGIMES:
            for seed in range(args.seeds):
                results.append(run_case(b, reg, seed, args))
            print(f"Complete: {results[-1]['system']}, {reg}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = {k: v for k, v in vars(args).items() if k != "out"}
    metadata.update({"horizons": list(range(1, H+1)), "n_features": 16, "numpy": np.__version__,
                     "primary_rank": 3, "metric": "mean absolute-scale conditional-mean RMSE over horizons"})
    args.out.joinpath("results.json").write_text(json.dumps(
        {"config": metadata, "results": results}, indent=2,
        default=lambda x: x.tolist(), allow_nan=False) + "\n")
    args.out.joinpath("README.md").write_text(report(results, args))


if __name__ == "__main__":
    main()
