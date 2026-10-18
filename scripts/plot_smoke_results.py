"""Render recorded benchmark JSON without rerunning optimization."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parents[1]
rows=json.loads((root/'reports/smoke/results.json').read_text())['runs']
names=['markov','gaussian_affine','triangular','logistic']
labels=['Markov, rank 3','Gaussian, affine span','Nonlinear triangular','Chaotic logistic']
fig,axes=plt.subplots(1,2,figsize=(11,4.3))
fig.patch.set_facecolor('#fafaf8')
colors=['#254d70','#ba6846','#638777']
for ax in axes: ax.set_facecolor('#fafaf8')
for j,(field,label) in enumerate([
    ('random_features_relative_rmse','Untrained + calibration'),
    ('neural_relative_rmse','Trained + calibration'),
    ('edmd_relative_rmse','EDMD reference')]):
    values=np.array([[r['horizon_metrics']['1'][field]*100 for r in rows if r['case']==n] for n in names])
    axes[0].bar(np.arange(4)+(j-1)*.23,values.mean(axis=1),width=.23,color=colors[j],label=label)
    axes[0].errorbar(np.arange(4)+(j-1)*.23,values.mean(axis=1),yerr=values.std(axis=1),fmt='none',color='#303030',lw=.8,capsize=2)
axes[0].set_xticks(range(4),labels,rotation=20,ha='right')
axes[0].set_ylabel('One-step relative RMSE (%)')
axes[0].set_title('Prediction: mean ± seed standard deviation')
axes[0].legend(fontsize=8,frameon=False)
for i,n in enumerate(names):
    values=np.array([[r['horizon_metrics'][str(h)]['neural_error_over_initial_rms']*100 for h in [1,5,10]] for r in rows if r['case']==n])
    axes[1].plot([1,5,10],values.mean(axis=0),marker='o',label=labels[i])
axes[1].set_xticks([1,5,10]); axes[1].set_xlabel('Prediction horizon (steps)')
axes[1].set_ylabel('Prediction error / initial-state RMS (%)')
axes[1].set_title('Calibrated neural operator: horizon dependence')
axes[1].legend(fontsize=8,frameon=False)
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.18); ax.set_axisbelow(True)
fig.suptitle('Koopman–NeuralSVD · bounded exploratory CPU experiments',fontsize=14)
fig.text(.5,.015,'Three seeds · 4,096 training pairs · at most 600 steps/model · stochastic targets are conditional means',ha='center',fontsize=8,color='#555555')
fig.tight_layout(rect=[0,.05,1,.94])
fig.savefig(root/'reports/smoke/comparison.svg',metadata={'Date':None})
fig.savefig(root/'reports/smoke/comparison.png',dpi=160)

svg=root/'reports/smoke/comparison.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
