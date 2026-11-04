"""Follow-up to nonstationary_pilot.py: select a forecast blend on fresh seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nonstationary_pilot import (Chain, WarpedCircle, H, REGIMES, regression_baselines,
                                 system_seed, validation_sample)
from koopman_svd.certification import certify_improvement
from koopman_svd.transport import TimeIndexedKoopmanSVD

WEIGHTS = np.linspace(0., 1., 9)


def run(bench, regime, seed, ntrain, ncal):
    """Select one weight for all horizons; never present the truth to the selector."""
    sys = (Chain, WarpedCircle)[bench](regime)
    training, cal, features = [np.random.default_rng(s) for s in system_seed(
        bench, regime, seed).spawn(3)]
    train = sys.paths(training, sys.starts(training, ntrain))
    W = features.normal(0, 1.3 / np.sqrt(train.shape[-1]), (train.shape[-1], 16))
    bias = features.uniform(-1, 1, 16)
    feat = lambda x: np.tanh(x @ W + bias)
    reference, mass = sys.reference()
    x, y, _ = validation_sample(sys, cal, ncal)
    xx = np.concatenate([reference, x])
    nr = len(reference)
    ols = regression_baselines(train, feat, sys.target)
    pool = ols["pooled EDMD"](xx)
    time_edmd = ols["time EDMD"](xx)
    transport = TimeIndexedKoopmanSVD.fit(train, rank=None, left_features=feat, right_features=feat)
    dynamic = np.clip(np.column_stack([transport.predict(
        xx, start=0, horizon=h, observable=sys.target).ravel() for h in range(1, H + 1)]), -1, 1)
    # Blend bounded endpoint predictions; all menu entries fixed before calibration.
    menu = (1 - WEIGHTS[:, None, None]) * pool[None] + WEIGHTS[:, None, None] * dynamic[None]
    preds = np.concatenate([menu, time_edmd[None]])
    pc = preds[:, nr:].transpose(0, 2, 1)
    # Selection by observable noisy MSE is valid here: the noise adds the same term to each model.
    # Equal horizon weights are predeclared and do not depend on the realized futures.
    scores = ((pc[:len(menu)] - y) ** 2).mean(axis=(1, 2))
    choice = int(np.argmin(scores))
    cpool = certify_improvement(pc, y, baseline=0, bounds=(-1, 1), delta=.025)
    ctime = certify_improvement(pc, y, baseline=len(menu), bounds=(-1, 1), delta=.025)
    ref_mse = np.einsum("mnh,n->mh", (preds[:, :nr] - sys.truth(reference)[None])**2, mass)
    return {
        "system": sys.name, "regime": regime, "seed": seed, "selected_time_weight": WEIGHTS[choice],
        "retained_ranks": [len(s.singular_values) for s in transport.steps],
        "population_rmse": {key: np.sqrt(ref_mse[index]) for key, index in
                            (("adaptive", choice), ("pooled", 0), ("time SVD", len(menu)-1),
                             ("time EDMD", len(menu)))},
        "selected_gain_lower_vs_pool": cpool.lower[choice],
        "selected_gain_lower_vs_time_EDMD": ctime.lower[choice],
        "whole_menu_covered": bool(np.all(cpool.lower <= ref_mse[0] - ref_mse + 1e-12)
                                       and np.all(ctime.lower <= ref_mse[-1] - ref_mse + 1e-12)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--train", type=int, default=1024)
    p.add_argument("--cal", type=int, default=4096)
    p.add_argument("--out", type=Path, default=Path("reports/nonstationary"))
    args = p.parse_args()
    if args.seeds < 1 or args.train < 2 or args.cal < 2:
        p.error("at least one seed and two starts per split required")
    rows = [run(i, reg, seed, args.train, args.cal)
            for i in range(2) for reg in REGIMES for seed in range(100, 100+args.seeds)]
    args.out.mkdir(exist_ok=True, parents=True)
    args.out.joinpath("pooling.json").write_text(json.dumps(
        {"config": {"train": args.train, "cal": args.cal, "weight_grid": WEIGHTS.tolist(),
                    "seeds": list(range(100, 100+args.seeds)), "numpy": np.__version__}, "results": rows},
        indent=2, default=lambda x: x.tolist(), allow_nan=False) + "\n")
    ls = [
        "# When should transition data be pooled?", "",
        "The first pilot showed opposite winners when only the initial distribution changes and when the "
        "transition law itself changes. This separately designed follow-up tests whether calibration data "
        "can distinguish them. It uses fresh seeds starting at 100; systems, observables and horizon range "
        "are unchanged. This remains an exploratory synthetic experiment.", "",
        f"The method blends the clipped pooled EDMD forecast (weight 0) and the clipped full available "
        "time-specific singular-function forecast (weight 1); the latter uses the same fixed tanh dictionary "
        f"and the correct cross-time bridges. Choose one weight from {len(WEIGHTS)} evenly spaced candidates "
        f"by minimizing noisy mean squared error across all five horizons on {args.cal} independent "
        f"calibration paths after fitting both endpoints on the same {args.train} training paths. "
        "The estimator uses neither exact conditional means nor a regime label. It does use a known time "
        "index for the local endpoint and needs validation rollouts from the deployment distribution. "
        "The selected weight is fixed across horizons. This is a forecasting ensemble; its weights are not "
        "singular values, and it does not improve the SVD itself.", "",
        "| System | Regime | Pooled | Time SVD | Time EDMD | Adaptive | Selected time weights (seed order) | "
        "Certified gains vs pool / vs time EDMD (of seed × horizon cases) |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for sys in ("finite chain", "warped circle"):
        for reg in REGIMES:
            group = [r for r in rows if r["system"] == sys and r["regime"] == reg]
            errors = [np.mean([r["population_rmse"][k] for r in group]) for k in
                      ("pooled", "time SVD", "time EDMD", "adaptive")]
            gain = [sum((r[k] > 0).sum() for r in group) for k in
                    ("selected_gain_lower_vs_pool", "selected_gain_lower_vs_time_EDMD")]
            ls.append(f"| {sys} | {reg} | " + " | ".join(f"{v:.4f}" for v in errors) + " | "
                      + str([float(r["selected_time_weight"]) for r in group]) + f" | {gain[0]} / {gain[1]} of {len(group)*H} |")
    ls.extend([
        "", "Errors are conditional-mean RMSE averaged over horizons 1–5 and seeds, in bounded observable "
        "units. A positive simultaneous lower bound means the selected model improves the observable MSE "
        "at that horizon; an inconclusive bound is not a declaration of equality. Two predeclared "
        "baseline comparisons split the 5% failure budget equally within each run. Their finite menus "
        "include all weights, so selection using the same calibration outcomes is allowed.",
        "", f"Both full comparison families covered their population contrasts in "
        f"{sum(r['whole_menu_covered'] for r in rows)}/{len(rows)} runs. Coverage is a diagnostic, not "
        "evidence of universal superiority. Exact finite-state risks are summed and circle risks use "
        "the periodic quadrature defined in the first pilot. There is no global guarantee over the full collection of runs.",
        "", "Forecast mixing is a conventional bias–variance choice. It can reduce the loss of making "
        "the wrong pooling decision; it can also have higher error than an endpoint that is appropriate "
        "for a particular regime. The menu and fresh seed family were fixed after examining the first pilot. "
        "These experiments do not establish novelty, robust controller performance or benefit on real data.",
        "", "```bash",
        f"OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python experiments/adaptive_pooling.py "
        f"--seeds {args.seeds} --train {args.train} --cal {args.cal}",
        "```", "",
    ])
    args.out.joinpath("pooling.md").write_text("\n".join(ls))


if __name__ == "__main__":
    main()
