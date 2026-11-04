"""EDMD and direct-regression controls for held-out pendulum schedules.

All covariates and moment matrices are fit from whole training trajectories. The
requested damping schedule is a known exogenous input. No target trajectories
are used to estimate target moments, means, or dictionaries. The returned
forecasts are conditional-mean candidates for the scalar observable cos(theta).
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import numpy as np


RIDGE = 1e-5
TIME_BANDWIDTH = 2.0          # transition intervals
DAMPING_BANDWIDTH = 0.4      # dimensionless damping coefficient
RANDOM_FEATURES = 16


def _input(a, d):
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[1] != d or not np.isfinite(a).all():
        raise ValueError("expected finite (starts, sensor_dim) inputs")
    return a


class Dictionary:
    """Frozen, seed-matched nonlinear dictionary containing the observable."""

    def __init__(self, d, seed, view, harmonic=False):
        self.d = d
        self.view = view
        self.harmonic = harmonic
        rng = np.random.default_rng(seed)
        self.W = rng.normal(size=(d, RANDOM_FEATURES)) / np.sqrt(d)
        self.bias = rng.normal(size=RANDOM_FEATURES)
        self.width = self(np.zeros((1, d))).shape[1]

    def __call__(self, x):
        x = _input(x, self.d)
        c = x[:, 0]
        if not self.harmonic:
            return np.column_stack((np.ones(len(x)), x, np.tanh(x @ self.W + self.bias)))
        s = x[:, 1]
        # Only observed sensor anchors. The structural comparator knows the
        # sensor calibration (including the full-view tanh velocity encoding).
        terms = [np.ones(len(x)), c, s]
        ck, sk = c, s
        for _ in range(2, 5):
            ck, sk = ck*c-sk*s, sk*c+ck*s
            terms.extend([ck, sk])
        if self.view == "full":
            v = 2*np.arctanh(np.clip(x[:, 2], -1+1e-12, 1-1e-12))
            terms.extend([v, v*v, v**3, c*v, s*v, c*v*v, s*v*v, terms[3]*v, terms[4]*v])
        return np.column_stack(terms)


class ReducedDictionary:
    """Exact constant and target plus PCA of remaining training residuals."""

    def __init__(self, base, train_obs, rank):
        self.base = base
        self.d = base.d
        if rank < 2:
            raise ValueError("observable-inclusive rank must be >= 2")
        raw = base(train_obs.reshape(-1, base.d))
        q = raw[:, :2]
        # Regress against [constant, cos(theta)] before extracting variance.
        self.offset = np.linalg.lstsq(q, raw[:, 2:], rcond=1e-10)[0]
        residual = raw[:, 2:] - q @ self.offset
        gram = residual.T @ residual/len(residual)
        vals, vectors = np.linalg.eigh(gram)
        keep = np.flatnonzero(vals > 1e-10*max(vals[-1], 1e-15))[::-1][:rank-2]
        self.projection = vectors[:, keep]
        self.width = 2+len(keep)

    def __call__(self, x):
        raw = self.base(x)
        return np.column_stack((raw[:, :2], (raw[:, 2:]-raw[:, :2] @ self.offset) @ self.projection))


def _weights(train_damping, target, t, method):
    if method == "pooled":
        w = np.ones(train_damping.shape)
    else:
        logw = -.5*((train_damping-target[t])/DAMPING_BANDWIDTH)**2
        if method == "time+damping":
            logw -= .5*((np.arange(train_damping.shape[1])-t)/TIME_BANDWIDTH)**2
        w = np.exp(logw-logw.max())
    return w/w.sum()


def _solve(G, C):
    """Affine ridge in normalized moment units; the intercept is unpenalized."""
    reg = RIDGE*np.eye(len(G)); reg[0, 0] = 0.
    return np.linalg.solve(G+reg, C)


@dataclass
class Moments:
    dictionary: object
    gram: np.ndarray
    cross: np.ndarray
    direct_cross: np.ndarray
    rank: int

    @classmethod
    def fit(cls, dictionary, train):
        s,n,tt,d = train.shape
        z = dictionary(train.reshape(-1,d)).reshape(s,n,tt,-1)
        x,y = z[:,:,:-1], z[:,:,1:]
        gram = np.einsum("snti,sntj->stij", x, x, optimize=True)/n
        cross = np.einsum("snti,sntj->stij", x, y, optimize=True)/n
        direct_cross = np.einsum("sni,snt->sti", z[:,:,0], train[:,:,1:,0], optimize=True)/n
        v=np.linalg.eigvalsh(gram.mean(axis=(0,1)))
        rank=int((v > 1e-10*max(v[-1],1e-15)).sum())
        return cls(dictionary, gram, cross, direct_cross, rank)


class Forecast:
    def __init__(self, name, mode, moments, damping):
        self.name, self.mode = name, mode
        self.moments, self.damping = moments, damping
        self.dictionary_width = 1 if mode == "persistence" else moments.dictionary.width
        self.state_size = self.dictionary_width
        if mode == "persistence":
            retained = 0
        elif mode == "direct":
            retained = damping.size + moments.gram[:,0].size + moments.direct_cross.size
        elif mode == "pooled":
            retained = self.state_size**2
        else:
            retained = damping.size + moments.gram.size + moments.cross.size
        self.metadata = {
            "name": name, "model_family": mode, "state_size": self.state_size,
            "dictionary_width": self.dictionary_width,
            "effective_rank": 1 if mode == "persistence" else moments.rank,
            "operator_storage_scalars": retained,
            "storage_definition": "Operator coefficients or sufficient statistics required for deployment; excludes feature map, unused shared caches and Python overhead.",
            "rank_definition": "Numerical rank of the average training dictionary Gram at relative cutoff 1e-10, not the transfer-operator rank.",
            "ridge": None if mode == "persistence" else RIDGE,
            "time_bandwidth": TIME_BANDWIDTH if mode == "time+damping" else None,
            "damping_bandwidth": DAMPING_BANDWIDTH if mode in ("direct","damping","time+damping") else None,
        }
        self._pooled = self._operator(damping[0], 0) if mode == "pooled" else None

    def _operator(self, target, t):
        w = _weights(self.damping, target, t, self.mode)
        g, c = (np.tensordot(w, a, axes=2) for a in (self.moments.gram,self.moments.cross))
        a = _solve(g, c)
        # Exact conditional expectation of the constant.
        a[:,0] = 0.; a[0,0] = 1.
        return a

    def predict(self, x0, damping):
        """Clipped forecasts (starts, horizons 1..H); no evaluation-path fitting."""
        x0 = _input(x0, self.moments.dictionary.d)
        target = np.asarray(damping, dtype=float)
        if target.ndim != 1 or not 1 <= len(target) <= self.damping.shape[1] or not np.isfinite(target).all():
            raise ValueError("invalid target damping schedule")
        if self.mode == "persistence":
            return np.repeat(np.clip(x0[:,0,None],-1,1),len(target),axis=1)
        z = self.moments.dictionary(x0)
        out = np.empty((len(x0),len(target)))
        for t in range(len(target)):
            if self.mode == "direct":
                # Direct regression from the same initial-state law as the query.
                # Nearness of prefixes is computed on exogenous covariates only.
                dist = ((self.damping[:,:t+1]-target[:t+1])**2).mean(axis=1)
                logw = -.5*dist/DAMPING_BANDWIDTH**2
                w=np.exp(logw-logw.max()); w/=w.sum()
                coef=_solve(np.tensordot(w,self.moments.gram[:,0],axes=1),
                            np.tensordot(w,self.moments.direct_cross[:,t],axes=1))
                out[:,t]=z @ coef
            else:
                a = self._pooled if self.mode == "pooled" else self._operator(target,t)
                z = z @ a
                out[:,t]=z[:,1]
        return np.clip(out,-1,1)


def fit_baselines(train_obs, train_damping, *, seed=0, matched_rank=4, view="full"):
    """Fit without seeing any validation/calibration/test trajectories.

    Input axes are (training schedule, path, time, sensor) and (schedule,
    transition). Every forecast receives the same initial sensor observation and
    future damping schedule; persistence and the unweighted pooled control
    intentionally ignore the schedule. A separate fit is needed for partial
    views, which never receive the unobserved velocity.
    """
    tr=np.asarray(train_obs, dtype=float)
    damping=np.array(train_damping, dtype=float, copy=True)
    if (tr.ndim!=4 or min(tr.shape)<1 or tr.shape[1]<2 or tr.shape[2]<2 or tr.shape[3]<3 or
            damping.shape!=(tr.shape[0],tr.shape[2]-1) or not np.isfinite(tr).all() or
            not np.isfinite(damping).all() or view not in ("full","partial")):
        raise ValueError("invalid training paths, damping array or observation view")
    started = perf_counter()
    raw=Dictionary(tr.shape[-1],seed,view)
    moments=Moments.fit(raw,tr)
    base_seconds = perf_counter()-started
    started = perf_counter()
    harmonic=Moments.fit(Dictionary(tr.shape[-1],seed,view,harmonic=True),tr)
    feature_fit_seconds = {id(moments): base_seconds, id(harmonic): perf_counter()-started}
    defs=[("persistence","persistence",moments),("edmd_pooled","pooled",moments),
          ("edmd_smoothed","time+damping",moments),("edmd_damping","damping",moments),
          ("direct_horizon","direct",moments),("edmd_harmonic","damping",harmonic)]
    for rank in dict.fromkeys([matched_rank,8]):
        started = perf_counter()
        compact=Moments.fit(ReducedDictionary(raw,tr,rank),tr)
        feature_fit_seconds[id(compact)] = perf_counter()-started
        defs.append((f"edmd_rank{rank}","damping",compact))
    models = {}
    for name,mode,m in defs:
        started = perf_counter()
        models[name] = Forecast(name,mode,m,damping)
        models[name].metadata["fit_seconds"] = (perf_counter()-started +
            (0. if mode == "persistence" else feature_fit_seconds[id(m)]))
        models[name].metadata["timing_definition"] = (
            "Elapsed construction time including the applicable feature/moment cache; "
            "the same cache is shared across some arms. Schedule-conditioned solves "
            "occur at forecasting time. This is not a hardware-independent cost.")
    return models
