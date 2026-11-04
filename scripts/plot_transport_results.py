"""Plot the recorded study, without refitting or selecting models on test data."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports"
plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "koopman-transport",
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def data(path):
    return json.loads((REPORT / path).read_text())


def main():
    g = data("gaussian-transport/smoothing.json")["summary"]
    rows = data("nonstationary/results.json")["results"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.9), layout="constrained")
    colors = ["#4e687e", "#a1563a", "#4e8078", "#283b64"]
    xs = np.arange(3)
    for j, method in enumerate(["pooled", "local", "partial", "smooth"]):
        a.bar(xs + (j-1.5)*.18,
              [g[k]["rmse"][method] for k in ["stationary", "shifted_homogeneous", "drifting"]],
              width=.18, color=colors[j], label=["Pooled", "Time-local", "Partial pool", "Time-neighbor"][j])
    a.set(xticks=xs, xticklabels=["Stationary", "Shifted initial law", "Drifting transition"],
          ylabel="Three-step conditional-mean RMSE", ylim=(0,.29),
          title="A. Sharing data helps when transitions are similar")
    a.legend(frameon=False, ncol=2, fontsize=8)
    eps = np.linspace(.05, .35, 151)
    accepted, oracle = [], []
    for tol in eps:
        valid, possible = [], []
        for row in rows:
            bound = np.sqrt(row["risk_upper"])
            truth = np.array([row["population_rmse_by_horizon"][f"rank {k}"] for k in row["ranks"]])
            valid.append(np.any(np.all(bound <= tol, axis=1)))
            possible.append(np.any(np.all(truth <= tol, axis=1)))
        accepted.append(np.mean(valid))
        oracle.append(np.mean(possible))
    b.step(eps, oracle, where="post", color=colors[0], linestyle="--", label="Exact-risk oracle")
    b.step(eps, accepted, where="post", color=colors[2], linewidth=2, label="Certified")
    b.axvline(.30, color="#999", lw=1, linestyle=":", label="Pilot tolerance")
    b.set(xlabel="Allowed RMSE (observable in [−1, 1])", ylabel="Fraction of 30 fits",
          ylim=(-.025,1.075), title="B. Certifying five steps depends on the tolerance")
    b.legend(frameon=False, fontsize=8, loc="lower right")
    svg = REPORT / "nonstationary" / "study.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
