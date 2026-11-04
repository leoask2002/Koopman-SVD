"""Regenerate the study figure from the frozen, committed numerical results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "reports/pendulum"
COLORS = {"classical":"#186150", "learned":"#b4512a", "upstream":"#52657c", "privileged":"#584986"}


def main():
    fits = json.loads((P/"results.json").read_text())["fits"]
    fig,(ax,bx) = plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for fam,label in [("classical","Time-smoothed EDMD"),("learned","Learned NeuralSVD"),
                      ("upstream","Upstream family"),("privileged","Known force-law form")]:
        curves=[]
        for fit in fits:
            key=fit["views"]["full"]["winners"][fam]
            paired=[fit["cases"]["full_A_"+s]["models"][key]["mse"] for s in ("test_smooth","test_switch")]
            curves.append(np.sqrt(np.maximum(0,np.mean(paired,axis=0))))
        ax.plot(np.arange(1,7),np.mean(curves,axis=0),marker="o",color=COLORS[fam],label=label)
    ax.set(title="A. Conditional-mean error on unseen damping schedules",ylabel="RMSE of cosine of angle",
           xlabel="Forecast horizon (steps of 0.25)",xticks=range(1,7),ylim=(0,.55))
    ax.legend(frameon=False,fontsize=8)
    cs=json.loads((P/"certificates.json").read_text())
    for j,name in enumerate(["persistence","classical","learned"]):
        rows=[c["threshold"][name] for c in cs]
        means=[np.mean([r[x+"_prefix"] for r in rows]) for x in ("eb","betting")]
        for i,(v,col) in enumerate(zip(means,["#8799a5","#186150"])):
            y=j+(i-.5)*.32
            bx.barh(y,v,height=.28,color=col,label=(["Empirical Bernstein","Fixed-threshold betting"][i] if j==0 else None))
            bx.text(v+.08,y,format(v,'.1f'),va='center',fontsize=8)
    bx.set(title="B. How far could the forecast be certified?", xlabel="Mean consecutive horizons accepted (of 6)",
           yticks=[0,1,2],yticklabels=["Persistence","Time-smoothed EDMD","Learned NeuralSVD"],xlim=(0,6))
    bx.legend(loc="lower right",frameon=False,fontsize=8)
    plt.rcParams.update({"svg.fonttype":"none","svg.hashsalt":"pendulum-transfer"})
    path=P/"comparison.svg"
    fig.savefig(path, metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
