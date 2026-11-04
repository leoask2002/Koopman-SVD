"""Fixed-budget stochastic pendulum comparison. Run from the repository root."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from koopman_svd.certification import certify_forecasts, certify_improvement
from pendulum_system import damping, draw_starts, simulate, observe, conditional_endpoints, reference_scores

SEEDS = tuple(range(71, 76))
H = 6
NTRAIN = NVAL = 256
NTEST, BRANCHES, NCAL = 512, 128, 4096
TRAIN = ("train_low", "train_high", "train_ramp", "train_step")
VAL = ("val_smooth", "val_switch")
TEST = ("test_smooth", "test_switch")
TOL, DELTA_CALL = 0.15, 0.05 / 12


def rng(seed, *coordinates):
    # No hashed string labels, shared RNG cursor, or ordering dependence.
    return np.random.default_rng(np.random.SeedSequence([seed, *coordinates]))


def paths(seed, labels, split, n):
    states, schedules = [], []
    for j, label in enumerate(labels):
        g = damping(label, H)
        stream = rng(seed, split, j)
        states.append(simulate(stream, draw_starts(stream, n), g))
        schedules.append(g)
    return np.stack(states), np.stack(schedules)


def plain(a):
    if isinstance(a, dict):
        return {str(k): plain(v) for k, v in a.items()}
    if isinstance(a, (list, tuple)):
        return [plain(i) for i in a]
    if isinstance(a, np.ndarray):
        return plain(a.tolist())
    if isinstance(a, np.generic):
        return a.item()
    return a


def predict(model, x, g):
    p = np.asarray(model.predict(x, g), dtype=float)
    if p.shape != (len(x), len(g)) or not np.isfinite(p).all():
        raise ValueError(f"invalid forecast for {type(model)}: {p.shape}")
    return np.clip(p, -1.0, 1.0)


def fit(seed, view, tr, gt, va, gv, official=True):
    from pendulum_baselines import fit_baselines
    from pendulum_neural import fit_neural
    models, fam, resources = {}, {}, {}

    def add(key, m, family, elapsed):
        if key in models:
            raise ValueError(f"duplicate model: {key}")
        models[key] = m
        fam[key] = family
        md = getattr(m, "metadata", {})
        md = md() if callable(md) else md
        resources[key] = dict(md)
        resources[key].update(fit_group_seconds=elapsed,
                              state_size=int(getattr(m, "state_size", resources[key].get("state_size", 0))),
                              dictionary_width=int(getattr(m, "dictionary_width", resources[key].get("dictionary_width", 0))))

    t = time.perf_counter()
    base = fit_baselines(tr, gt, seed=seed, matched_rank=4, view=view)
    elapsed = time.perf_counter()-t
    for key, m in base.items():
        add(key, m, "classical", elapsed)
    for rank in (4, 8):
        t = time.perf_counter()
        learned = fit_neural(tr, gt, va, gv, seed=seed, matched_rank=rank, steps=64, width=16)
        elapsed = time.perf_counter()-t
        for label, m in learned.items():
            family = {"neural": "learned", "frozen": "frozen", "sensor_spectral": "classical"}[label]
            add(f"{label}_r{rank}", m, family, elapsed)
        if official:
            from pendulum_upstream import fit_official
            t = time.perf_counter()
            upstream = fit_official(tr, gt, va, gv, seed=seed, matched_rank=rank, steps=64, width=16)
            elapsed = time.perf_counter()-t
            for label, m in upstream.items():
                add(f"{label}_r{rank}", m, "upstream", elapsed)
    if view == "full":
        from pendulum_mechanics import fit_mechanics
        t = time.perf_counter()
        m = fit_mechanics(tr, gt)
        add("privileged_mechanics", m, "privileged", time.perf_counter()-t)

    val = {key: np.mean([(predict(m, scene[:, 0], g)[:, -1] - scene[:, -1, 0])**2
                          for scene, g in zip(va, gv)]) for key, m in models.items()}
    winners = {f: min((k for k in models if fam[k] == f), key=lambda k: (val[k], k)) for f in sorted(set(fam.values()))}
    return models, winners, val, resources, fam


def assess(seed, idx, label, view, sensor, models, winners, auditdir):
    t = damping(label, H)
    x = draw_starts(rng(seed, 30, idx), NTEST)
    xcal = draw_starts(rng(seed, 40, idx), NCAL)
    # Share state/forcing streams across A vs B; hidden velocity is redrawn
    # across branches when conditioning only on angle (not on its latent start).
    ref = conditional_endpoints(rng(seed, 31, idx), x, t, view=view, branches=BRANCHES)
    yc = conditional_endpoints(rng(seed, 41, idx), xcal, t, view=view, branches=2)
    obs, co = observe(x, view, sensor), observe(xcal, view, sensor)
    names = list(models)
    cold = {}
    for k,m in models.items():
        t0 = time.perf_counter()
        predict(m, obs[:128], t)
        cold[k] = (time.perf_counter() - t0)*1000
    pp = np.stack([predict(m, obs, t) for m in models.values()])
    pc = np.stack([predict(m, co, t) for m in models.values()])
    # Timed deployment uses identical starts and horizon counts, warmed caches.
    # Report the cost of producing new predictions (schedule may already be cached).
    latency = {}
    for k, m in models.items():
        xx = obs[:128]
        predict(m, xx, t)
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            predict(m, xx, t)
            times.append((time.perf_counter()-t0)*1e3)
        latency[k] = float(np.median(times))

    score = reference_scores(pp, ref)
    cc = pc.transpose(0, 2, 1)
    absolute = certify_forecasts(cc, yc[:, 0].T, yc[:, 1].T, bounds=(-1, 1), delta=DELTA_CALL)
    contrast = certify_improvement(cc, yc[:, 0].T, baseline=names.index(winners["classical"]),
                                   bounds=(-1, 1), delta=DELTA_CALL)
    prefixes = np.logical_and.accumulate(absolute.upper <= TOL**2, axis=1).sum(axis=1)
    rows = {k: {field: score[field][j] for field in ("raw_mse","mse","rmse","mse_se")} |
               {"abs_upper": absolute.upper[j], "gain_lower": contrast.lower[j],
                "gain_estimate": contrast.estimate[j], "prefix": prefixes[j], "latency_ms_128": latency[k], "first_query_ms_128": cold[k]}
            for j, k in enumerate(names)}
    if auditdir and view == "full" and sensor == "A":
        auditdir.mkdir(exist_ok=True, parents=True)
        np.savez_compressed(auditdir/f"seed{seed}_{label}.npz", y=yc[:, 0].T, y2=yc[:, 1].T, test_ref=ref,
                            cal_predictions=pc, test_predictions=pp, names=np.array(names),
                            classic=np.array(winners["classical"]), learned=np.array(winners["learned"]))
    return {"reference_noise_mse": score["reference_noise_floor"], "models": rows}


def run(seed, official, auditdir):
    tr, gt = paths(seed, TRAIN, 10, NTRAIN)
    va, gv = paths(seed, VAL, 20, NVAL)
    row = dict(seed=seed, views={}, cases={})
    for view, sensors in (("full", ("A", "B")), ("partial", ("A",))):
        train, val = observe(tr, view, "A"), observe(va, view, "A")
        models, winners, scores, resources, fam = fit(seed, view, train, gt, val, gv, official)
        row["views"][view] = dict(winners=winners, validation_h6_mse=scores,
                                  resources=resources, family=fam)
        for sensor in sensors:
            for idx, label in enumerate(TEST):
                row["cases"][f"{view}_{sensor}_{label}"] = assess(seed, idx, label, view, sensor,
                                                                 models, winners, auditdir)
    return plain(row)


def boot_paired(x, y):
    """Differences/ratios of mean seed-wise RMSE, resampling whole paired seeds."""
    x, y = np.array(x), np.array(y)
    ii = rng(8732, 4).integers(0, len(x), size=(10000, len(x)))
    a, b = x[ii].mean(axis=1), y[ii].mean(axis=1)
    ci = lambda z: np.quantile(z, [.025, .975]).tolist()
    return dict(mean=float(np.mean(x-y)), interval=ci(a-b), ratio=float(x.mean()/y.mean()),
                ratio_interval=ci(a/b))


def summarize(rows):
    families = list(rows[0]["views"]["full"]["winners"])
    primary = {f: [] for f in families}
    primary_cost = {f: [] for f in families}
    primary_size = {f: [] for f in families}
    prefix = {f: [] for f in families}
    for row in rows:
        info = row["views"]["full"]
        for family, name in info["winners"].items():
            stats = [row["cases"][f"full_A_{label}"]["models"][name] for label in TEST]
            primary[family].append(np.sqrt(max(0., np.mean([r["mse"][-1] for r in stats]))))
            primary_cost[family].append(np.mean([r["latency_ms_128"] for r in stats]))
            primary_size[family].append(info["resources"][name]["state_size"])
            prefix[family].extend(r["prefix"] for r in stats)
    means = {f: np.mean(v) for f, v in primary.items()}
    q = boot_paired(primary["learned"], primary["classical"])
    # Direct-horizon models can have no propagated (recurrent) state.
    adequate_dims = all(i > 0 and i >= 2*j for i,j in zip(primary_size["classical"],primary_size["learned"]))
    adequate_dims = adequate_dims and all("direct" not in row["views"]["full"]["winners"]["classical"] for row in rows)
    cost_ratio = np.median(primary_cost["learned"]) / np.median(primary_cost["classical"])
    accuracy = q["ratio"] < .90
    compression = q["ratio"] <= 1.05 and adequate_dims and cost_ratio < 1
    cert = min(prefix["learned"]) == H
    return plain(dict(primary_rmse=means, per_seed=primary, learned_minus_classical=q,
                      primary_latency_ms={f: np.median(v) for f,v in primary_cost.items()},
                      primary_dimensions=primary_size, prefix=prefix, latency_ratio=cost_ratio,
                      gate=dict(accuracy=accuracy, compression=compression, horizon=cert,
                                scale=bool((accuracy or compression) and cert))))


def report(result):
    rows, s = result["fits"], result["summary"]
    comparison = s["learned_minus_classical"]
    seeds = len(rows)
    link = "[fixed protocol](protocol.md)"
    lines = ["# Stochastic pendulum: unseen damping schedules and sensor changes", "",
             "This is a bounded physical **simulation**, not a hardware experiment. The forecasts target the conditional mean of cos(angle). The simulator defines a discrete process; no inference about the exact continuous-time generator or closed-loop control is made. See the "+link+" and [machine-readable results](results.json).",
             "", "## Primary result", "",
             "For each fit we selected one candidate in each family by horizon-six loss on independent validation schedules. Primary scores are the mean of per-seed RMSEs on full sensor A at 1.5 time units: each seed averages corrected MSE equally over two genuinely held-out damping schedules before taking a square root. Lower is better.",
             "", "| Validation-selected family | Primary RMSE | Warm median ms / 128 starts | Propagated/model dimensions by fit |",
             "|---|---:|---:|---|"]
    for fam, mean in s["primary_rmse"].items():
        lines.append(f"| {fam} | {mean:.5f} | {s['primary_latency_ms'][fam]:.3f} | {s['primary_dimensions'][fam]} |")
    lines.extend(["",f"Learned minus classical: **{comparison['mean']:+.5f}** (paired 95% seed-bootstrap interval [{comparison['interval'][0]:+.5f}, {comparison['interval'][1]:+.5f}]); RMSE ratio **{comparison['ratio']:.3f}** [{comparison['ratio_interval'][0]:.3f}, {comparison['ratio_interval'][1]:.3f}]. Intervals are descriptive only: {seeds} independent fits, two specific unseen schedules, common evaluation draws across candidates. The comparison uses validation winners, not a posteriori test winners.",
                  "", "## Stress tests and certificates", "",
                  "Entries are mean horizon-six corrected RMSE across training seeds. The prefix is the mean count of consecutive certified horizons for the validation-selected learned model at absolute RMSE tolerance 0.15. A positive gain lower bound compares learned against the validation-selected classical model at horizon six; absence of a positive bound is inconclusive.",
                  "", "| Observation | Test damping | Classical | Learned | Frozen | Official | Privileged | Learned prefix / 6 | Positive gain bounds |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for view, sensor in (("full","A"),("full","B"),("partial","A")):
        for label in TEST:
            key = f"{view}_{sensor}_{label}"
            vals = {f: [] for f in s["primary_rmse"]}
            prefixes, gains = [], []
            for row in rows:
                case = row["cases"][key]["models"]
                w = row["views"][view]["winners"]
                for f, k in w.items():
                    vals[f].append(case[k]["rmse"][-1])
                prefixes.append(case[w["learned"]]["prefix"])
                gains.append(case[w["learned"]]["gain_lower"][-1]>0)
            columns = [f"{np.mean(vals[f]):.4f}" if vals.get(f) else "—"
                       for f in ("classical","learned","frozen","upstream","privileged")]
            lines.append(f"| {view} {sensor} | {label} | "+" | ".join(columns)
                         +f" | {np.mean(prefixes):.1f} | {sum(gains)}/{seeds} |")
    # Expose a strong nuisance-invariant control on the stress arm; avoid
    # confusing a relative gain over a failing validation winner with robustness.
    bs = lambda name,label: np.mean([r["cases"]["full_B_"+label]["models"][name]["rmse"][-1] for r in rows])
    lines += ["", f"On held-out nuisance map B the fixed harmonic EDMD control, which uses the unchanged physical anchors, scored {bs('edmd_harmonic',TEST[0]):.4f}/{bs('edmd_harmonic',TEST[1]):.4f} on smooth/switch. Even persistence scored {bs('persistence',TEST[0]):.4f}/{bs('persistence',TEST[1]):.4f}. A positive gain over a model selected on A but failing on B is therefore not a demonstration of sensor robustness."]
    lines.extend(["", "## Fixed candidate inspection", "",
                  "Every training-map-A candidate below is scored at horizon six on the two primary schedules, whether or not validation selected it. Selection counts use only the validation schedules and only competitors within the same family.", "",
                  "| Candidate | Family | Primary RMSE | Selected fits | Coordinates | Feature width | Warm ms / 128 |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for name in rows[0]["views"]["full"]["family"]:
        fam = rows[0]["views"]["full"]["family"][name]
        values, speeds, wins = [], [], 0
        for row in rows:
            stats = [row["cases"][f"full_A_{label}"]["models"][name] for label in TEST]
            values.append(np.sqrt(max(0., np.mean([r["mse"][-1] for r in stats]))))
            speeds.append(np.mean([r["latency_ms_128"] for r in stats]))
            wins += row["views"]["full"]["winners"][fam] == name
        res = rows[0]["views"]["full"]["resources"][name]
        lines.append(f"| {name} | {fam} | {np.mean(values):.5f} | {wins}/{seeds} | {res['state_size']} | {res['dictionary_width']} | {np.median(speeds):.3f} |")
    noise = np.mean([row["cases"][f"full_A_{label}"]["reference_noise_mse"][-1] for row in rows for label in TEST])
    unc = np.mean([row["cases"][f"full_A_{label}"]["models"][row["views"]["full"]["winners"]["classical"]]["mse_se"][-1]
                   for row in rows for label in TEST])
    lines.append(f"\nOn the primary endpoint the average finite-reference variance contribution to raw MSE was {noise:.6g} (equivalent RMSE scale {np.sqrt(noise):.5f}). The mean independent-test-start standard error of the corrected MSE for the selected classical model was {unc:.6g}. This is simulation uncertainty, not across-fit uncertainty. Unlike trajectory RMSE against single noisy endpoints, the corrected target measures conditional-mean forecasting error.")
    lines += ["","A→B changes only five redundant nuisance sensor channels. Three calibrated informative anchors are unchanged; there is no fitting or validation on B. The partial view has no velocity channel; the latent initial velocity is redrawn independently on each reference/calibration branch, so the reference conditions only on available information. The mechanics comparator is privileged (decodes the calibrated channels and assumes the governing form). Direct regression dimensions describe a dictionary rather than propagated rank; the compression gate does not treat a direct winner as a larger recurrent state.",
              "", "## Cost, evidence and decision", "",
              f"Gate: accuracy (>10% lower primary RMSE) **{s['gate']['accuracy']}**; compression (≤5% excess, ≥2× rank reduction and faster forecasting) **{s['gate']['compression']}**; all six horizons certified on both primary schedules for all fits **{s['gate']['horizon']}**. Overall advance-to-scale decision: **{s['gate']['scale']}**.",
              "",f"Per-fit model dimensions and warm-start latencies (median of five one-thread runs for 128 starts and all six steps) are reported, along with first-query timings, fit-group times, candidate-specific storage proxies and trainable parameters where defined. Model size, number of features and runtime differ; timing depends on the machine and cached coefficient availability. No large model was trained.",
              "",f"Per seed: 4×{NTRAIN} training and 2×{NVAL} validation paths; {NCAL} new calibration starts and {NTEST} independent test starts with {BRANCHES} independent continuations per test start/schedule. Signed variance-corrected MSE, its independent-start standard error, raw MSE, and finite-reference variance contribution are recorded for each horizon. Clipping negative corrected MSE to zero for the displayed RMSE can matter for very accurate predictors. The absolute certificates and one-sided gain bounds split a 5% familywise budget over six scenarios and two types of bound per fitted seed. They cover average risk for the stated deployment starts; they do not certify individual trajectories under arbitrary shifts.",
              "", "## Reproduction", "",
              "From the repository root, install the package and test extra. The full study also needs the [official adapter instructions](../../docs/pendulum-upstream.md). Then run:",
              "", "```sh", "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python experiments/pendulum_study.py", "```", "",
              "The main JSON records all per-seed candidate metrics, validation choices, budgets and schedules. Optional scratch audit arrays contain individual calibration forecasts and independent test references; they are not needed to read the report. No method may use them for further tuning and still claim these are fresh confirmatory tests."]
    lines += ["", "## Additional diagnostics", "",
              "The near-zero mechanics score needs more conditional branches than the generic forecasts. See the [independent high-precision audit](mechanics.md) and [discretization checks](physics.md). Warm latency is a property of these implementations and their caching policies; it is not an inherent advantage of NeuralSVD. The full-A risk family uses 21 candidates and six horizons. Its empirical Bernstein range term alone adds approximately 0.0314 in squared-error units at 4,096 starts, larger than the target tolerance squared (0.0225); signed residual means and sample variances determine the actual bound. This explains why even a good forecaster can have an uninformative guarantee here."]
    return "\n".join(lines)+"\n"


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reports/pendulum"))
    parser.add_argument("--audit-dir", type=Path, default=Path("runs/pendulum-audit"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--without-official", action="store_true", help="Run core-only panel; distinct from the reported official comparison.")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        print(f"seed {seed}",flush=True)
        rows.append(run(seed, not args.without_official, args.audit_dir))
    result = dict(config={"seeds":args.seeds,"H":H,"train_per_schedule":NTRAIN,"val_per_schedule":NVAL,
                          "cal_starts":NCAL,"test_starts":NTEST,"test_branches":BRANCHES,
                          "tolerance":TOL,"delta_per_family":DELTA_CALL,"official_included":not args.without_official,
                          "schedules":{k:damping(k,H).tolist() for k in TRAIN+VAL+TEST},
                          "versions":{"python":platform.python_version(),"numpy":np.__version__,"torch":torch.__version__}},
                  summary=summarize(rows),fits=rows)
    (args.out/"results.json").write_text(json.dumps(plain(result),separators=(",",":"),allow_nan=False)+"\n")
    (args.out/"results.md").write_text(report(result))
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
