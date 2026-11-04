"""Small Gaussian reference study of partial pooling and nonstationary reversal.

Run from repository root: python experiments/gaussian_transport.py
See reports/gaussian-transport/protocol.md for the prespecified comparisons.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

T = 9
HORIZONS = (1, 3, 6)
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)  # weight on time-local fit
REGIMES = ("stationary", "shifted_homogeneous", "drifting")


@dataclass
class Dynamics:
    A: np.ndarray
    b: np.ndarray
    Q: np.ndarray
    m0: np.ndarray
    C0: np.ndarray

    def marginals(self):
        m, c = [self.m0], [self.C0]
        for a, b, q in zip(self.A, self.b, self.Q):
            m.append(a @ m[-1] + b)
            cov = a @ c[-1] @ a.T + q
            c.append((cov + cov.T) / 2)
        return np.array(m), np.array(c)

    def maps(self):
        m, c = self.marginals()
        rev = np.stack([np.linalg.solve(c[t + 1], a @ c[t]).T for t, a in enumerate(self.A)])
        return self.A, self.b, rev, m[:-1] - np.einsum("tij,tj->ti", rev, m[1:])


def system(regime):
    d = 2
    a = np.repeat(np.array([[[0.72, 0.20], [-0.15, 0.65]]]), T, axis=0)
    b = np.zeros((T, d))
    q = np.repeat(np.array([[[0.12, 0.025], [0.025, 0.09]]]), T, axis=0)
    # Solve C = ACA^T + Q. Row-major vectorization is used with identical factors.
    c0 = np.linalg.solve(np.eye(d * d) - np.kron(a[0], a[0]), q[0].ravel()).reshape(d, d)
    m0 = np.zeros(d)
    if regime in ("shifted_homogeneous", "drifting"):
        m0 = np.array([1.6, -1.25])
        c0 = np.array([[1.4, 0.40], [0.40, 0.42]])
    if regime == "drifting":
        angles = 2 * np.pi * np.arange(T) / T
        a[:, 0, 0] += 0.18 * np.cos(angles)
        a[:, 0, 1] += 0.23 * np.sin(angles)
        a[:, 1, 0] += 0.19 * np.sin(angles + 0.5)
        a[:, 1, 1] -= 0.13 * np.cos(angles)
        b = np.stack([0.14 * np.sin(angles), 0.10 * np.cos(angles)], axis=1)
    if regime not in REGIMES:
        raise ValueError("unknown regime")
    return Dynamics(a, b, q, m0, c0)


def simulate(law, rng, n):
    states = [rng.multivariate_normal(law.m0, law.C0, n)]
    for a, b, q in zip(law.A, law.b, law.Q):
        states.append(states[-1] @ a.T + b + rng.multivariate_normal(np.zeros(len(b)), q, n))
    return np.stack(states, axis=1)


def ols(x, y):
    x, y = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])
    beta = np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]
    return beta[:-1].T, beta[-1]


def fit_models(paths):
    n, td, d = paths.shape
    x, y = paths[:, :-1], paths[:, 1:]
    ap, bp = ols(x, y)
    local = [ols(x[:, t], y[:, t]) for t in range(td - 1)]
    al, bl = map(np.array, zip(*local))
    m0 = x[:, 0].mean(axis=0)
    c0 = (x[:, 0] - m0).T @ (x[:, 0] - m0) / n
    fits = {}
    for w in WEIGHTS:
        a, b = (1 - w) * ap + w * al, (1 - w) * bp + w * bl
        res = y - np.einsum("tij,ntj->nti", a, x) - b
        qt = np.einsum("nti,ntj->tij", res, res) / n
        # Blend local and pooled residual moments, including systematic local
        # intercept mismatch; at w=1 this is the standard local Gaussian MLE.
        q = (1 - w) * qt.mean(axis=0) + w * qt
        fits[w] = Dynamics(a, b, q, m0, c0).maps()
    # Explicit negative control: fit and reuse one affine predictor independently
    # in each direction across the whole trajectory.
    ar, br = ols(y, x)
    fixed = (fits[0.0][0], fits[0.0][1],
             np.repeat(ar[None], td - 1, axis=0), np.repeat(br[None], td - 1, axis=0))
    return fits, fixed


def compositions(maps, h):
    """For each t, return (forward, reverse) maps connecting times t and t+h."""
    a, b, ar, br = maps
    n, d = len(a) - h + 1, a.shape[-1]
    def product(aa, bb, backwards):
        m = np.repeat(np.eye(d)[None], n, axis=0)
        v = np.zeros((n, d))
        for j in (reversed(range(h)) if backwards else range(h)):
            aj, bj = aa[j:j+n], bb[j:j+n]
            m, v = aj @ m, np.einsum("tij,tj->ti", aj, v) + bj
        return m, v
    return (product(a, b, False), product(ar, br, True))


def validation_loss(maps, paths):
    total = 0.0
    for h in HORIZONS:
        for j, (a, b) in enumerate(compositions(maps, h)):
            x, y = (paths[:, :-h], paths[:, h:]) if j == 0 else (paths[:, h:], paths[:, :-h])
            predicted = np.einsum("tij,ntj->nti", a, x) + b
            total += np.mean((predicted - y) ** 2)
    return total / (2 * len(HORIZONS))


def score(maps, reference, m, c):
    out = {}
    for h in HORIZONS:
        errs = []
        for j, ((a, b), (a0, b0)) in enumerate(zip(compositions(maps, h), compositions(reference, h))):
            # Exact E||delta_A X + delta_b||^2/d under the proper start marginal.
            da, db = a - a0, b - b0
            start = slice(None, -h) if j == 0 else slice(h, None)
            bias = np.einsum("tij,tj->ti", da, m[start]) + db
            var = da @ c[start] @ da.swapaxes(1, 2)
            errs.append(float((np.sum(bias ** 2, axis=1) + np.trace(var, axis1=1, axis2=2)).mean() / a.shape[-1]))
        out[str(h)] = dict(forward=np.sqrt(errs[0]), reverse=np.sqrt(errs[1]), joint=np.sqrt(np.mean(errs)))
    return out


def run_seed(law, seed):
    tr, va = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    train, valid = simulate(law, tr, 32), simulate(law, va, 96)
    fits, fixed = fit_models(train)
    losses = {w: validation_loss(f, valid) for w, f in fits.items()}
    best = min(WEIGHTS, key=losses.get)
    endpoint = min((0.0, 1.0), key=losses.get)
    reference = law.maps()
    m, c = law.marginals()
    by_w = {w: score(f, reference, m, c) for w, f in fits.items()}
    return dict(seed=seed, selected_w=best, endpoint_w=endpoint,
                validation_mse=[losses[w] for w in WEIGHTS],
                scores=dict(pooled=by_w[0.0], local=by_w[1.0], half=by_w[0.5],
                            adaptive=by_w[best], endpoint_choice=by_w[endpoint],
                            fixed_reverse=score(fixed, reference, m, c)))


def summarize(rows):
    methods = tuple(rows[0]["scores"])
    def values(method, h, direction="joint"):
        return np.array([r["scores"][method][str(h)][direction] for r in rows])
    means = {method: {str(h): {s: float(values(method, h, s).mean()) for s in ("joint", "forward", "reverse")}
                             for h in HORIZONS} for method in methods}
    # Same seed-pair resampling for every contrast. Negative differences favor adaptive.
    rng = np.random.default_rng(246810)
    bootstrap = rng.integers(len(rows), size=(10_000, len(rows)))
    diffs = {}
    for method in methods:
        if method == "adaptive":
            continue
        paired = values("adaptive", 3) - values(method, 3)
        low, high = np.quantile(paired[bootstrap].mean(axis=1), [0.025, 0.975])
        diffs[method] = dict(mean=float(paired.mean()), lo=float(low), hi=float(high))
    return dict(mean_seed_rmse=means, h3_paired_adaptive_minus_comparator=diffs,
                selected_weight_counts={str(w): sum(r["selected_w"] == w for r in rows) for w in WEIGHTS})


def render(data):
    lines = ["# Partial pooling for changing Gaussian state distributions", "",
             "[Protocol](protocol.md) · [per-seed results](results.json) · [code](../../experiments/gaussian_transport.py)",
             "", "See also the [exploratory time-neighbor comparison](smoothing.md). Run `python experiments/gaussian_transport.py` from the repository root (NumPy only). "
             "No neural network was trained. This is an exact affine reference, not evidence of a neural advantage.",
             "", "## Results", "",
             "Entries are mean **three-step conditional-mean RMSE** across independent fits, in common synthetic state units. "
             "Each seed is scored by exact population integration across both directions, coordinates and all time origins. "
             "Lower is better. This is not RMSE against noisy observations.", "",
             "| System | Fixed reverse | Pooled, changing reverse | Local | Fixed half | Endpoint choice | Adaptive |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for regime, summary in data["summary"].items():
        means = summary["mean_seed_rmse"]
        lines.append("| " + " | ".join([regime] + [f"{means[m]['3']['joint']:.4f}" for m in
                     ("fixed_reverse", "pooled", "local", "half", "endpoint_choice", "adaptive")]) + " |")
    lines += ["", "Paired differences below are adaptive minus comparator, drifting system at h=3, the prespecified "
              "primary endpoint. Intervals are descriptive 95% percentile bootstrap intervals over 30 independent fitting seeds "
              "(10,000 resamples); they do not establish transfer to other systems or correct for multiple comparisons.", "",
              "| Comparator | Paired mean RMSE difference | Interval |", "|---|---:|---:|"]
    for method, v in data["summary"]["drifting"]["h3_paired_adaptive_minus_comparator"].items():
        lines.append(f"| {method} | {v['mean']:+.5f} | [{v['lo']:+.5f}, {v['hi']:+.5f}] |")
    lines += ["", "Selection frequencies (columns are weights on the local fit):", "",
              "| System | 0 | 0.25 | 0.5 | 0.75 | 1 |", "|---|---:|---:|---:|---:|---:|"]
    for regime, su in data["summary"].items():
        lines.append("| " + " | ".join([regime] + [str(su["selected_weight_counts"][str(w)]) for w in WEIGHTS]) + " |")
    lines += ["", "## Estimator and interpretation", "",
              "For each step, least-squares estimates of the affine forward map are shrunk toward one pooled affine fit. "
              "The same weight blends time-local residual second moments with their pooled mean. An initial sample mean "
              "and covariance are propagated through this fitted Gaussian model. The reverse conditional slope at time t "
              "is C_t A_tᵀ C_{t+1}⁻¹, with intercept m_t − B_t m_{t+1}. Reversing an interval composes these slopes "
              "in reverse time order. Each candidate is fit on training trajectories; the validation score uses observed "
              "pairs, never the analytic reference. Exact truth is used only for evaluation.",
              "", "In a time-homogeneous system the forward model can share all parameters while its correct reverse map "
              "changes as the state distribution evolves. The fixed reverse diagnostic reuses a pooled backwards regression "
              "and should therefore not be the main comparator. When dynamics drift, local fits avoid homogeneous model bias "
              "but can have high estimation variance; the comparison against endpoint selection isolates the value of "
              "intermediate shrinkage weights. Fixed half-pooling is a strong control for whether validation adds value. "
              "Adaptive pooling is not novel by itself, and the preferred weight is sensitive to sample size, drift shape "
              "and prediction tasks.",
              "", "All three regimes are exactly closed in the constant-plus-linear observable space. An affine Koopman "
              "model at full rank can represent the conditional means without neural approximation; the benchmark tests "
              "estimation, time reversal and pooling, not the full infinite-dimensional singular spectrum. The Gaussian "
              "observables are unbounded, so the bounded-observable coverage theorem from the other pilot does not apply.",
              "", "See the JSON for forward and reverse results at horizons 1, 3, and 6 plus every seed, all validation "
              "scores, and prespecified design constants. No generator identification, single-trajectory mixing theorem, "
              "nonlinear generalization result, or controller claim is inferred."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reports/gaussian-transport"))
    args = parser.parse_args()
    rows = {r: [run_seed(system(r), seed) for seed in range(30)] for r in REGIMES}
    data = dict(design=dict(n_train=32, n_validation=96, seeds=list(range(30)), n_steps=T,
                            horizons=HORIZONS, local_weights=WEIGHTS, primary="drifting: joint h=3"),
                summary={r: summarize(v) for r, v in rows.items()}, by_seed=rows)
    args.out.mkdir(exist_ok=True, parents=True)
    (args.out / "results.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    (args.out / "README.md").write_text(render(data))
    print(render(data))


if __name__ == "__main__":
    main()
