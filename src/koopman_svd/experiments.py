"""Bounded CPU experiments with independent train/validation/test samples.

No external data, GPUs, or long-running jobs. The reported spectra are finite
learned-subspace estimates, never a claim about an entire deterministic operator.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .model import NeuralKoopmanSVD


def polynomial_features(x, degree=2):
    x = np.asarray(x)
    if x.shape[1] == 1:
        # Shifted Chebyshev basis is better conditioned than monomials on [0,1].
        return np.polynomial.chebyshev.chebvander(2*x[:, 0]-1, degree)
    if x.shape[1] == 2 and degree == 2:
        a, b = x.T
        return np.column_stack([np.ones(len(x)), a, b, a*a, a*b, b*b])
    return np.column_stack([np.ones(len(x)), x])


class EDMD:
    """Explicit reference baseline; this does not train NeuralSVD."""
    def __init__(self, degree=1):
        self.degree = degree

    def fit(self, x, y):
        px, py = self.features(x), self.features(y)
        self.operator = np.linalg.lstsq(px, py, rcond=1e-10)[0]
        self.decoder = np.linalg.lstsq(px, x, rcond=1e-10)[0]
        return self

    def features(self, x):
        if self.degree == 1:
            return np.column_stack([np.ones(len(x)), x])
        return polynomial_features(x, self.degree)

    def predict(self, x, steps=1):
        return self.features(x) @ np.linalg.matrix_power(self.operator, steps) @ self.decoder


def relative_rmse(pred, truth):
    denominator = np.mean(np.square(truth))
    return float(np.sqrt(np.mean(np.square(pred-truth))/max(denominator, 1e-30)))


def stationary_distribution(p):
    a = np.vstack([p.T-np.eye(len(p)), np.ones(len(p))])
    return np.linalg.lstsq(a, np.r_[np.zeros(len(p)), 1.], rcond=None)[0]


def markov_pairs(p, pi, n, rng):
    i = rng.choice(len(p), n, p=pi)
    j = np.array([rng.choice(len(p), p=p[k]) for k in i])
    return np.eye(len(p))[i], np.eye(len(p))[j]


def triangular(x):
    return np.column_stack([0.8*x[:, 0], 0.7*x[:, 1]+(0.8**2-0.7)*x[:, 0]**2])


def logistic(x):
    return 4*x*(1-x)


def cases(seed, n, markov_rank=3):
    rng = np.random.default_rng(seed+1000)
    # Column-state convention Y = A X + noise, row samples Y = X A.T.
    a = np.array([[.65, .20, 0.], [0., .50, .12], [0., 0., .30]])
    q = np.eye(3)-a@a.T
    def gaussian(m):
        x = rng.normal(size=(m, 3))
        return x, x@a.T+rng.normal(size=(m, 3))@np.linalg.cholesky(q).T
    p = np.array([[.65,.25,.08,.02],[.05,.65,.25,.05],
                  [.10,.05,.65,.20],[.20,.10,.05,.65]])
    pi = stationary_distribution(p)
    weighted = np.sqrt(pi)[:,None]*p/np.sqrt(pi)[None,:]
    def finite(m):
        return markov_pairs(p, pi, m, rng)
    def nonlinear(m):
        x = rng.uniform(-1, 1, (m, 2))
        return x, triangular(x)
    def chaotic(m):
        x = np.sin(np.pi*rng.random((m, 1)))**2
        return x, logistic(x)
    definitions = [
        ('markov', finite, 4, markov_rank, (16,), 1, np.linalg.svd(weighted,compute_uv=False)[:markov_rank]),
        ('gaussian_affine', gaussian, 3, 4, (), 1, np.r_[1., np.linalg.svd(a,compute_uv=False)]),
        ('triangular', nonlinear, 2, 6, (32,32), 2, None),
        ('logistic', chaotic, 1, 6, (32,32), 8, None),
    ]
    for name, generate, dim, rank, hidden, degree, spectrum in definitions:
        train, validation, test = generate(n), generate(max(n//4,256)), generate(max(n//2,512))
        yield dict(name=name,train=train,validation=validation,test=test,dim=dim,rank=rank,
                   hidden=hidden,degree=degree,spectrum=spectrum,A=a,P=p,pi=pi)


def exact_future(case, x, horizon):
    if case['name']=='markov':
        return x@np.linalg.matrix_power(case['P'],horizon)
    if case['name']=='gaussian_affine':
        return x@np.linalg.matrix_power(case['A'].T,horizon)
    step = triangular if case['name']=='triangular' else logistic
    for _ in range(horizon):
        x = step(x)
    return x


def serializable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k):serializable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [serializable(v) for v in value]
    return value


def run_case(case, seed, steps, max_seconds, batch_size=256):
    x,y = case['train']
    model = NeuralKoopmanSVD(case['dim'],case['rank'],hidden_dims=case['hidden'],seed=seed)
    random_model = NeuralKoopmanSVD(case['dim'],case['rank'],hidden_dims=case['hidden'],seed=seed)
    random_model.fit(x,y,steps=0)
    start=time.monotonic()
    model.fit(x,y,validation_data=case['validation'],steps=steps,batch_size=batch_size,
              lr=0.003,patience=20,max_seconds=max_seconds)
    baseline=EDMD(case['degree']).fit(x,y)
    # All choices and fitting have finished before test metrics are computed.
    xt,yt=case['test']
    if case['name']=='markov':
        xt=np.eye(4)  # Enumerate every state instead of a random test approximation.
    metrics={}
    for h in [1,5,10]:
        truth=exact_future(case,xt,h)
        metrics[str(h)]={
            'neural_relative_rmse':relative_rmse(model.predict(xt,steps=h),truth),
            'edmd_relative_rmse':relative_rmse(baseline.predict(xt,steps=h),truth),
            'persistence_relative_rmse':relative_rmse(xt,truth),
            'neural_rmse':float(np.sqrt(np.mean((model.predict(xt,steps=h)-truth)**2))),
            'neural_error_over_initial_rms':float(np.sqrt(np.mean((model.predict(xt,steps=h)-truth)**2)/np.mean(xt**2))),
            'random_features_relative_rmse':relative_rmse(random_model.predict(xt,steps=h),truth),
        }
        if case['name']=='markov':
            pi=case['pi']
            u,s,vh=np.linalg.svd(np.sqrt(pi)[:,None]*case['P']/np.sqrt(pi)[None,:])
            r=case['rank']
            oracle=(u[:,:r]*s[:r])@vh[:r]
            oracle=oracle/np.sqrt(pi)[:,None]*np.sqrt(pi)[None,:]
            metrics[str(h)]['exact_rank_truncation_relative_rmse']=relative_rmse(
                xt@np.linalg.matrix_power(oracle,h),truth)
    raw_prediction=model.raw_functions(xt,side='left') @ (model.raw_functions(y,side='right').T@y/len(y))
    result=dict(case=case['name'],seed=seed,steps_requested=steps,batch_size=batch_size,rank=case['rank'],
                hidden_dims=case['hidden'],n_train=len(x),n_validation=len(case['validation'][0]),
                n_test=len(xt),seconds=time.monotonic()-start,
                fit=model.fit_summary_,calibration=model.calibration_summary_,
                training_history=model.history_,
                raw_singular_values=model.raw_singular_values_,
                raw_neural_one_step_relative_rmse=relative_rmse(raw_prediction,exact_future(case,xt,1)),
                calibrated_singular_values=model.singular_values_,
                true_reference_singular_values=case['spectrum'],
                horizon_metrics=metrics,test_diagnostics=model.diagnostics(*case['test']))
    if case['spectrum'] is not None:
        padded=np.pad(np.asarray(model.singular_values_), (0,len(case['spectrum'])-len(model.singular_values_)))
        result['calibrated_spectrum_relative_error']=float(np.linalg.norm(
            padded-case['spectrum'])/np.linalg.norm(case['spectrum']))
        result['raw_spectrum_relative_error']=float(np.linalg.norm(
            np.asarray(model.raw_singular_values_)-case['spectrum'])/np.linalg.norm(case['spectrum']))
        result['raw_spectrum_descending_order']=bool(np.all(np.diff(model.raw_singular_values_)<=0))
    if case['name']=='gaussian_affine':
        result['noisy_test_relative_rmse']=relative_rmse(model.predict(case['test'][0]),yt)
        result['irreducible_noise_mean_squared_norm']=float(np.trace(np.eye(3)-case['A']@case['A'].T))
    return serializable(result)


def write_report(payload, output):
    output.mkdir(parents=True,exist_ok=True)
    (output/'results.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    lines=['# Bounded CPU experiment results','',
           'These are small synthetic checks, not a full training campaign or a reproduction of the published benchmarks.',
           'Every dataset uses independent training, validation and test draws; test data never choose a checkpoint.',
           'These benchmark designs were explored iteratively; they are exploratory results, not a pristine confirmatory holdout study.',
           'Relative RMSE is sqrt(mean squared prediction error / mean squared ground truth).',
           'Stochastic forecasts are scored against exact conditional expectations, not individual noisy outcomes.',
           'The Gaussian reference is restricted to affine observables. Deterministic cases report prediction only.',
           'Post-training calibration performs a small dense SVD; the training objective itself does not.',
           '', '| Case | Seed | Raw spectrum error | Calibrated spectrum error | Neural 1-step | EDMD 1-step | Neural 10-step | EDMD 10-step |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda x: '—' if x is None else f'{100*x:.3f}%'
    for r in payload['runs']:
        one,ten=r['horizon_metrics']['1'],r['horizon_metrics']['10']
        lines.append(f"| {r['case']} | {r['seed']} | {fmt(r.get('raw_spectrum_relative_error'))} | {fmt(r.get('calibrated_spectrum_relative_error'))} | {fmt(one['neural_relative_rmse'])} | {fmt(one['edmd_relative_rmse'])} | {fmt(ten['neural_relative_rmse'])} | {fmt(ten['edmd_relative_rmse'])} |")
    lines += ['', '## Interpretation', '',
      '- Long-horizon relative errors can be large when the true conditional mean decays toward zero. Absolute RMSE and error normalized by the initial-state RMS are also recorded in results.json.',
      '- The random-feature baseline has identical seeded networks with zero optimization steps and the same post-fit calibration. This isolates learned-subspace gains from post-fit linear algebra.',
      '- The affine case is an orientation/calibration sanity check. A complete affine span makes calibrated prediction an EDMD-type solution; it cannot establish a benefit from neural optimization.',
      '- A rank-three Markov model compresses a four-state chain; a rank truncation and sampling floor remain. The exact truncated reference is recorded in JSON. With --markov-rank 4 the full state space is retained.',
      '- Raw norm-based spectra need near-orthogonal converged factors. Calibrated spectra characterize the learned subspaces and can improve even when raw optimization remains incomplete.',
      '- Triangular polynomial EDMD has a known exact closure for the observed state. The general MLP is deliberately not given that closure.',
      '- Chaotic logistic forecasts and infinite-dimensional deterministic singular spectra have different limitations. Good one-step error does not guarantee long-horizon accuracy.',
      '- One or a few seeds do not establish statistical significance. No large datasets, GPU runs, control policies or continuous-time generator identification are validated.',
      '', '## Reproduction metadata', '', '```json',json.dumps(payload['metadata'],indent=2),'```','']
    lines += ['## Neural training ablation','',
      '| Case | Seed | Raw neural 1-step error | Random features + calibration | Trained features + calibration | 10-step error / initial RMS |',
      '|---|---:|---:|---:|---:|---:|']
    for r in payload['runs']:
        one,ten=r['horizon_metrics']['1'],r['horizon_metrics']['10']
        lines.append(f"| {r['case']} | {r['seed']} | {fmt(r['raw_neural_one_step_relative_rmse'])} | {fmt(one['random_features_relative_rmse'])} | {fmt(one['neural_relative_rmse'])} | {fmt(ten['neural_error_over_initial_rms'])} |")
    (output/'results.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true',help='100 steps, 1024 training pairs per case')
    parser.add_argument('--steps',type=int,default=None)
    parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--cases',nargs='+',choices=['markov','gaussian_affine','triangular','logistic'])
    parser.add_argument('--max-seconds',type=float,default=30.,help='Per model optimization wall-time cap')
    parser.add_argument('--markov-rank',type=int,choices=[3,4],default=3)
    parser.add_argument('--output',type=Path,default=Path('runs/smoke'))
    args=parser.parse_args(argv)
    steps=args.steps if args.steps is not None else (100 if args.quick else 600)
    if not 1<=steps<=5000 or args.max_seconds<=0 or args.batch_size<1:
        parser.error('steps must be 1..5000; max-seconds and batch-size must be positive')
    torch.set_num_threads(1)
    metadata=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
                  device='cpu',threads=1,seeds=args.seeds,steps=steps,n_train=1024 if args.quick else 4096,
                  max_seconds_per_model=args.max_seconds,
                  markov_rank=args.markov_rank,
                  batch_size=args.batch_size,
                  command='python -m koopman_svd.experiments '+ ' '.join(__import__('sys').argv[1:]))
    try: metadata['kooplearn']=importlib.metadata.version('kooplearn')
    except importlib.metadata.PackageNotFoundError: pass
    payload={'metadata':metadata,'runs':[]}
    for seed in args.seeds:
        for case in cases(seed,metadata['n_train'],args.markov_rank):
            if args.cases and case['name'] not in args.cases: continue
            result=run_case(case,seed,steps,args.max_seconds,args.batch_size)
            payload['runs'].append(result)
            write_report(payload,args.output)
            print(f"{case['name']} seed={seed}: one-step relative RMSE="
                  f"{result['horizon_metrics']['1']['neural_relative_rmse']:.5f}",flush=True)
    print(f'Results written to {args.output}',flush=True)


if __name__=='__main__':
    main()
