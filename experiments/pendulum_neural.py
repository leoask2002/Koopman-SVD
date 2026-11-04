"""Matched learned/frozen two-sided SVD for unseen prescribed damping schedules.

Only *training* state paths contribute moments. For each requested horizon h,
kernel-weight entire training ensembles by damping-prefix similarity. A single
set of weights applies to all steps of that prefix: the right and next left
spaces then share exactly the same surrogate intermediate marginal. It need
not coincide with the unknown held-out marginal. No target rollouts, including
unlabelled target pairs, are required. Learned and frozen use the same
calibration, bridge, decoding, input channels, rank and bandwidth.
"""
from __future__ import annotations

import time

import numpy as np
from koopman_svd.model import NeuralKoopmanSVD


PREFIX_BANDWIDTH = 0.4
DEFAULT_UPDATES = 64
GRAM_RCOND = 1e-9


def _check(paths, damping):
    x = np.asarray(paths, dtype=float)
    d = np.asarray(damping, dtype=float)
    if x.ndim != 4 or min(x.shape) < 1 or x.shape[2] < 2 or d.shape != (x.shape[0], x.shape[2]-1):
        raise ValueError("paths need (schedule, trajectory, T+1, sensors), damping (schedule,T)")
    if not (np.isfinite(x).all() and np.isfinite(d).all()):
        raise ValueError("non-finite paths or damping")
    return x, d


def _whitener(c):
    e, v = np.linalg.eigh(0.5 * (c + c.T))
    keep = e > max(np.finfo(float).tiny, GRAM_RCOND * max(float(e[-1]), 0.))
    return v[:, keep] / np.sqrt(e[keep])


class PrefixSpectralForecast:
    """A conditional finite SVD with sufficient statistics, not a fitted EDMD head."""

    def __init__(self, paths, damping, rank, model, name, *, bandwidth=PREFIX_BANDWIDTH):
        paths, self.damping = _check(paths, damping)
        self.damping = self.damping.copy()
        self.model = model
        self.name = name
        self.input_dim = paths.shape[-1]
        self.bandwidth = float(bandwidth)
        if rank < 1 or not np.isfinite(self.bandwidth) or self.bandwidth <= 0:
            raise ValueError("positive rank and finite positive damping bandwidth required")
        self.state_size = int(rank)
        self.dictionary_width = paths.shape[-1] + 1 if model is None else model.rank
        self._coeff = {}
        s, n, times, dim = paths.shape
        f = self._features(paths.reshape(-1, dim), "left").reshape(s,n,times,-1)
        g = self._features(paths.reshape(-1, dim), "right").reshape(s,n,times,-1)
        y = paths[...,0]
        self.stats = {
            "f": f.mean(axis=1), "g": g.mean(axis=1), "y": y.mean(axis=1),
            "ff": np.einsum("snti,sntj->stij", f,f)/n,
            "gg": np.einsum("snti,sntj->stij", g,g)/n,
            "fg1": np.einsum("snti,sntj->stij", f[:,:,:-1],g[:,:,1:])/n,
            "gf": np.einsum("snti,sntj->stij", g,f)/n,
            "gy": np.einsum("snti,snt->sti", g,y)/n,
        }
        par = 0 if model is None else sum(p.numel() for net in (model.source_net_, model.target_net_)
                                         for p in net.parameters())
        arrays = list(self.stats.values()) + [self.damping]
        # Deployment payload excluding Python overhead: moment tensors, schedules,
        # both neural maps and both sides' fixed normalization vectors. The optional
        # per-query coefficient cache is small and excluded; training-only checkpoint
        # metadata and unused original global calibration tensors are not needed.
        self.metadata = {
            "dictionary_width": int(self.dictionary_width), "state_size": self.state_size,
            "trainable_parameters": int(par) if name == "neural" else 0,
            "stored_network_parameters": int(par), "prefix_bandwidth": self.bandwidth,
            "model_bytes": int(sum(a.nbytes for a in arrays) + 8*(par+(4*dim if model else 0))),
            "size_definition": "deployment numeric payload; excludes Python overhead and optional coefficient cache",
            "calibration": "training-path damping-prefix Gaussian mixture; same-state left/right overlaps",
        }
        self.fit_seconds = 0.0

    def _features(self, x, side):
        if self.model is None:
            return x
        # Constant is pinned analytically in the conditional calibration below.
        return np.asarray(self.model.raw_functions(x, side=side))[:, 1:]

    def _prefix_coeff(self, schedule):
        h = len(schedule)
        distance = np.mean((self.damping[:, :h] - schedule[None])**2, axis=1)
        w = np.exp(-(distance-distance.min())/(2*self.bandwidth**2))
        w /= w.sum()
        m = {k: np.tensordot(w,v,axes=(0,0)) for k,v in self.stats.items()}
        next_left = None
        coeff = None
        for t in range(h-1,-1,-1):
            mu, nu = m["f"][t], m["g"][t+1]
            lw = _whitener(m["ff"][t] - np.outer(mu,mu))
            rw = _whitener(m["gg"][t+1] - np.outer(nu,nu))
            cross = m["fg1"][t] - np.outer(mu,nu)
            u, singular, vt = np.linalg.svd(lw.T @ cross @ rw,full_matrices=False)
            count = min(self.state_size - 1, len(singular))
            l, r = lw @ u[:,:count], rw @ vt[:count].T
            if coeff is None:
                const = m["y"][t+1]
                coeff = r.T @ (m["gy"][t+1] - nu*const)
            else:
                # E_{shared intermediate marginal}[right_t * left_(t+1)^T].
                overlap = m["gf"][t+1] - np.outer(nu, m["f"][t+1])
                coeff = r.T @ overlap @ next_left @ coeff
            coeff = singular[:count] * coeff
            next_left = l
        beta = l @ coeff
        return np.r_[const - mu @ beta, beta]

    def predict(self, x0, damping_schedule):
        x = np.asarray(x0, dtype=float)
        damp = np.asarray(damping_schedule, dtype=float)
        if damp.ndim != 1 or damp.size == 0 or damp.size > self.damping.shape[1] or not np.isfinite(damp).all():
            raise ValueError("invalid future damping schedule")
        if x.ndim != 2 or x.shape[1] != self.input_dim or not np.isfinite(x).all():
            raise ValueError("invalid initial sensor states")
        features = self._features(x,"left")
        key = tuple(damp)
        if key not in self._coeff:
            self._coeff[key] = np.column_stack([self._prefix_coeff(damp[:h+1]) for h in range(len(damp))])
        coeff = self._coeff[key]
        return np.clip(features @ coeff[1:] + coeff[:1], -1., 1.)


def fit_neural(train_obs, train_damping, val_obs, val_damping, *, seed=0, matched_rank=4,
               steps=DEFAULT_UPDATES, width=16):
    """Fit two matched feature arms plus an unoptimized raw sensor spectral control.

    Validation schedule paths are used only for the nested-loss checkpoint.
    Observables and operator moments are fitted on the original training paths.
    Exposing the future damping schedule is consistent with all controlled arms.
    """
    train, damping = _check(train_obs, train_damping)
    val, _ = _check(val_obs,val_damping)
    if train.shape[2:] != val.shape[2:]:
        raise ValueError("training/validation time and sensor shapes do not match")
    d = train.shape[-1]
    x, y = train[:,:,:-1].reshape(-1,d), train[:,:,1:].reshape(-1,d)
    xv, yv = val[:,:,:-1].reshape(-1,d), val[:,:,1:].reshape(-1,d)
    out = {}
    for name, updates in (("frozen",0), ("neural",steps), ("sensor_spectral",0)):
        tic = time.perf_counter()
        model = None
        if name != "sensor_spectral":
            model = NeuralKoopmanSVD(input_dim=d, rank=matched_rank, hidden_dims=(width,),seed=int(seed))
            model.fit(x,y,validation_data=(xv,yv),steps=updates,batch_size=128,
                      validation_every=8,patience=None,max_seconds=None)
        f = PrefixSpectralForecast(train,damping,matched_rank,model,name)
        f.fit_seconds = time.perf_counter()-tic
        f.metadata.update({"fit_seconds":f.fit_seconds,"steps":int(updates),
                           "selected_step":model.fit_summary_["best_step"] if model is not None else 0})
        out[name] = f
    return out


def fit_neural_models(train_paths, train_schedules, validation_paths, validation_schedules,
                      *, seed=0, rank=4, steps=DEFAULT_UPDATES, width=16):
    return fit_neural(train_paths, train_schedules, validation_paths, validation_schedules,
                      seed=seed, matched_rank=rank, steps=steps, width=width)
