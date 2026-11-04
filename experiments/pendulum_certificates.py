"""Evaluate the locked secondary checks on full-map-A study predictions (no refits)."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from koopman_svd.certification import (cap_forecast, certify_capped_updates, certify_forecasts,
                                       certify_improvement, certify_rmse_threshold)

CAPS = np.array([.10, .30, 2.])
SEEDS = range(71,76)
SCHEDULES = ("test_smooth", "test_switch")
LABELS = ("persistence", "classical", "learned")


def evaluate(path):
    with np.load(path, allow_pickle=False) as data:
        keys = list(data["names"])
        idx = [keys.index(k) for k in ("persistence", data["classic"].item(), data["learned"].item())]
        cal = data["cal_predictions"][idx].transpose(0,2,1)
        te = data["test_predictions"][idx].transpose(0,2,1)
        y, y2, ref = data["y"], data["y2"], data["test_ref"]
    mu = ref.mean(axis=1).T
    noise = ref.var(axis=1, ddof=1).T/ref.shape[1]
    eb = certify_forecasts(cal,y,y2,bounds=(-1,1),delta=.05/4)
    bets = certify_rmse_threshold(cal,y,y2,bounds=(-1,1),tolerance=.15,delta=.05/4)
    result = {"case":path.stem, "threshold":{}, "updates":[]}
    for j,key in enumerate(LABELS):
        result["threshold"][key]=dict(
            eb_prefix=int(np.logical_and.accumulate(eb.upper[j] < .15**2).sum()),
            betting_prefix=int(np.logical_and.accumulate(bets.accepted[j]).sum()),
            betting_log_e=bets.log_e_value[j].tolist(),risk_upper=eb.upper[j].tolist(),
            test_signed_mse=(((te[j]-mu)**2-noise).mean(axis=-1)).tolist())
    for j in (0,1):
        raw = np.repeat(cal[j+1][None], len(CAPS), axis=0)
        pc = np.stack([cap_forecast(cal[j], row, cap=c, bounds=(-1,1)) for c,row in zip(CAPS,raw)])
        full = certify_improvement(np.concatenate([cal[j:j+1],pc]), y, baseline=0,bounds=(-1,1),delta=.05/8)
        sharp = certify_capped_updates(raw,y,baseline_forecast=cal[j],caps=CAPS,bounds=(-1,1),delta=.05/8)
        for i,c in enumerate(CAPS):
            pt = cap_forecast(te[j],te[j+1],cap=c,bounds=(-1,1))
            result["updates"].append(dict(pair=f"{LABELS[j]} -> {LABELS[j+1]}",cap=float(c),
                    generic=full.lower[i+1].tolist(),restricted=sharp.lower[i+1].tolist(),
                    test_signed_mse=(((pt-mu)**2-noise).mean(axis=-1)).tolist(),
                    test_gain=(((te[j]-mu)**2-(pt-mu)**2).mean(axis=-1)).tolist()))
    return result


def main():
    ap=ArgumentParser()
    ap.add_argument("--audit",type=Path,default=Path("runs/pendulum-audit"))
    ap.add_argument("--out",type=Path,default=Path("reports/pendulum"))
    args=ap.parse_args()
    rows=[evaluate(args.audit/f"seed{s}_{g}.npz") for s in SEEDS for g in SCHEDULES]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"certificates.json").write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n")
    lines=["# What can 4,096 independent starts certify on a continuous state space?","",
        "Secondary checks on the same held-out full-map-A forecasts and stochastic outcomes. "
        "Each case uses 8,192 continuations (two per start). All models, budgets, threshold and caps were fixed before this analysis; see the "
        "[protocol](capped-protocol.md), [implementations and proofs](../../docs/certification.md), "
        "and [per-case evidence](certificates.json). This does not replace the primary scale-up gate.","",
        "## Fixed RMSE tolerance of 0.15","",
        "| Forecast | Empirical Bernstein: mean prefix | Betting: mean prefix | "
        "All six steps accepted: EB / betting (of 10 fits × schedules) |",
        "|---|---:|---:|---:|"]
    for key in LABELS:
        a=np.array([[r["threshold"][key][m+"_prefix"] for m in ("eb","betting")] for r in rows])
        lines.append(f"| {key} | {a[:,0].mean():.1f} | {a[:,1].mean():.1f} | "
                     f"{(a[:,0]==6).sum()} / {(a[:,1]==6).sum()} |")
    lines.extend(["","The procedures use identical samples, risk targets and finite candidate menus. "
        "They split a separate auxiliary 5% error allocation equally across the two schedules and two "
        "procedures of each fitted seed. The betting test answers only the single fixed threshold. "
        "It cannot be used to report an arbitrary tolerance interval without additional analysis.","",
        "## Capped observable updates","",
        "| Baseline → proposal | Maximum change | Positive lower bounds: generic / capped (of 60) | Mean h6 RMSE of capped forecast |",
        "|---|---:|---:|---:|"])
    for k,t in enumerate(rows[0]["updates"]):
        us=[r["updates"][k] for r in rows]
        g=sum((np.array(u["generic"])>0).sum() for u in us)
        s=sum((np.array(u["restricted"])>0).sum() for u in us)
        rmse=np.mean([np.sqrt(max(0,u["test_signed_mse"][-1])) for u in us])
        lines.append(f"| {t['pair']} | {t['cap']:.2f} | {g} / {s} | {rmse:.4f} |")
    lines.extend(["","Each cell certifies the **capped** prediction actually deployed. A full-range cap "
        "leaves the original prediction unchanged. Smaller caps reduce the largest possible score "
        "difference, but can reduce the gain as well. Positive bounds count model-fit × schedule × "
        "horizon cells; these are correlated, not independent replications. The two pairs, two schedules "
        "and two bound methods split a separate per-seed 5% family budget; caps and horizons are "
        "simultaneous in each call. Finite-support or observed-extrema assumptions are not used. "
        "A nonpositive lower bound is inconclusive.","",
        "Reproduce the per-start prediction cache with the main study command (see [results](results.md)); "
        "then run `python experiments/pendulum_certificates.py`. No further training is involved."])
    (args.out/"certificates.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
