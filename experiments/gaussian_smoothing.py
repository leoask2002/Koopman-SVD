"""Exploratory time-neighbor pooling for the Gaussian transport pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .gaussian_transport import REGIMES, Dynamics, fit_models, score, simulate, system, validation_loss
except ImportError:  # running directly from the repository root
    from gaussian_transport import REGIMES, Dynamics, fit_models, score, simulate, system, validation_loss

BANDWIDTHS = (0.0, 0.5, 1.0, 2.0, float("inf"))


def label(h):
    return "pooled" if np.isinf(h) else str(h)


def local_smooth(paths, bandwidth):
    """Use neighboring training times (no wrapping) to estimate the law at t."""
    x, y = paths[:, :-1], paths[:, 1:]
    n, k, d = x.shape
    xf = np.concatenate([x, np.ones((n, k, 1))], axis=2).reshape(-1, d + 1)
    yf = y.reshape(-1, d)
    a, b, q = [], [], []
    for t in range(k):
        if bandwidth == 0:
            wt = (np.arange(k) == t).astype(float)
        else:
            wt = np.exp(-0.5 * ((np.arange(k) - t) / bandwidth) ** 2)
        w = np.tile(wt, n)
        sw = np.sqrt(w[:, None])
        beta = np.linalg.lstsq(sw * xf, sw * yf, rcond=None)[0]
        res = yf - xf @ beta
        a.append(beta[:-1].T)
        b.append(beta[-1])
        q.append((sw * res).T @ (sw * res) / w.sum())
    m0 = x[:, 0].mean(axis=0)
    c0 = (x[:, 0] - m0).T @ (x[:, 0] - m0) / n
    return Dynamics(np.array(a), np.array(b), np.array(q), m0, c0).maps()


def seed_study(regime, seed):
    law = system(regime)
    tr, va = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    train, valid = simulate(law, tr, 32), simulate(law, va, 96)
    loc = {h: local_smooth(train, h) for h in BANDWIDTHS}
    shrink, _ = fit_models(train)
    best_h = min(BANDWIDTHS, key=lambda h: validation_loss(loc[h], valid))
    best_w = min(shrink, key=lambda w: validation_loss(shrink[w], valid))
    ms, cs = law.marginals()
    def risk(f):
        return score(f, law.maps(), ms, cs)
    return dict(seed=seed, selected_bandwidth=label(best_h), selected_local_weight=best_w,
                smooth=risk(loc[best_h]), partial=risk(shrink[best_w]),
                pooled=risk(shrink[0.0]), local=risk(shrink[1.0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reports/gaussian-transport"))
    out = parser.parse_args().out
    out.mkdir(exist_ok=True, parents=True)
    lines = ["# Temporal smoothing: exploratory stronger baseline", "",
             "[Timing and protocol](protocol.md) · [code](../../experiments/gaussian_smoothing.py) · [per-seed results](smoothing.json).",
             "Reproduce: `python experiments/gaussian_smoothing.py` from the repository root.", "",
             "After the primary Gaussian study, we evaluated a second class of time-dependent Gaussian models: "
             "fit each affine transition by weighting nearby training times with a Gaussian kernel. Select the global "
             "bandwidth on validation trajectories. Compare to validation-selected shrinkage of time-local fits toward "
             "one global fit. Both use exactly the same data, independent validation, and time-aware Gaussian reversal.",
             "", "| Regime | Pooled | Local | Global/local shrinkage | Time-neighbor smoothing | Smoothing − shrinkage (paired 95% interval) |",
             "|---|---:|---:|---:|---:|---:|"]
    report = {"bandwidths": [label(h) for h in BANDWIDTHS], "rows": {}, "summary": {}}
    idx = np.random.default_rng(246810).integers(30, size=(10_000, 30))
    for regime in REGIMES:
        rows = [seed_study(regime, i) for i in range(30)]
        values = {key: np.array([r[key]["3"]["joint"] for r in rows])
                  for key in ("pooled", "local", "partial", "smooth")}
        diff = values["smooth"] - values["partial"]
        lo, hi = np.quantile(diff[idx].mean(axis=1), [0.025, 0.975])
        tally = {label(h): sum(r["selected_bandwidth"] == label(h) for r in rows) for h in BANDWIDTHS}
        lines.append("| " + " | ".join([regime] + [f"{v.mean():.4f}" for v in values.values()] +
            [f"{diff.mean():+.5f} [{lo:+.5f}, {hi:+.5f}]"]) + " |")
        report["rows"][regime] = rows
        report["summary"][regime] = dict(rmse={k: float(v.mean()) for k, v in values.items()},
                                        paired_difference=dict(mean=float(diff.mean()), lo=float(lo), hi=float(hi)),
                                        selected_bandwidth_counts=tally)
    lines.extend(["", "RMSE is exact population three-step conditional-mean error as in [the primary results](README.md). "
                  "The intervals describe variation over 30 independent training/validation seeds of these synthetic laws. "
                  "This exploratory Gaussian comparison is conditional on smoothly varying coefficients; sharp switching, "
                  "unobserved time or irregular sampling deserve separate tests. Neither method is evidence of a neural "
                  "advantage. The bounded-output risk certificate elsewhere in this project does not cover Gaussian coordinates."])
    (out / "smoothing.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (out / "smoothing.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
