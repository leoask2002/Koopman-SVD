"""Optional, commit-pinned Jeong et al. pendulum comparator; no import-time downloads.

See docs/pendulum-upstream.md for installation, provenance and differences from
the paper's experiments. Setting KOOPMAN_UPSTREAM to a clean checkout opts in.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import os
import subprocess
import sys
import time

import numpy as np
import torch

try:
    from .pendulum_neural import PrefixSpectralForecast, _check
except ImportError:  # direct invocation from experiments/
    from pendulum_neural import PrefixSpectralForecast, _check

REVISION = "d76194d5b7e62d973edb2a226f2a54f4662b48cb"


@lru_cache(maxsize=1)
def _upstream(checkout):
    path = Path(checkout).resolve()
    rev = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if rev != REVISION:
        raise ValueError(f"Expected upstream revision {REVISION}; found {rev}")
    status = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain",
                                      "--untracked-files=all", "--", "koopmansvd"], text=True)
    if status.strip():
        raise ValueError("Upstream koopmansvd source differs from pinned checkout")
    sys.path.insert(0, str(path))
    try:
        import koopmansvd
        if Path(koopmansvd.__file__).resolve() != path / "koopmansvd/__init__.py":
            raise ImportError("A different koopmansvd installation was already imported")
        from koopmansvd.models import KoopmanPLModule, NestedLoRALoss
        from koopmansvd.models.encoders import CenteringWrapper, MLPEncoder
        from koopmansvd.models.inference import KoopmanEstimator
    finally:
        sys.path.remove(str(path))
    return KoopmanPLModule, NestedLoRALoss, CenteringWrapper, MLPEncoder, KoopmanEstimator


class _Features:
    def __init__(self, module, xm, xs, ym, ys, rank):
        self.source_net_, self.target_net_ = module.encoder, module.lagged_encoder
        self.norm = {"left": (xm, xs), "right": (ym, ys)}
        self.rank = rank

    @torch.inference_mode()
    def raw_functions(self, x, *, side):
        mean, scale = self.norm[side]
        net = self.source_net_ if side == "left" else self.target_net_
        return net((torch.as_tensor(x, dtype=mean.dtype) - mean) / scale).numpy()


class _NativeForecast:
    def __init__(self, features, estimator, name, metadata):
        self.features, self.estimator, self.name = features, estimator, name
        self.state_size = self.dictionary_width = features.rank
        self.fit_seconds = metadata["fit_seconds"]
        self.metadata = dict(metadata)
        # Minimal arrays used by the requested native forecast, excluding cached
        # training-only data, both networks already counted in metadata.
        if name == "upstream_cca":
            arrays = (estimator.alignments["f"], estimator.alignments["g"],
                      estimator.operators["ali"]["fwd"]["K"], estimator.stats["T_gy"])
        else:
            ops = estimator.operators["raw"]["f"]["fwd"]
            arrays = (ops["U"], ops["Q"], estimator.projections["f"])
        self.metadata["model_bytes"] += int(sum(a.nbytes for a in arrays))
        self.metadata["mode"] = name

    def predict(self, x0, damping):
        schedule = np.asarray(damping, dtype=float)
        if schedule.ndim != 1 or not len(schedule) or not np.isfinite(schedule).all():
            raise ValueError("expected a finite one-dimensional future schedule")
        f = self.features.raw_functions(x0, side="left")
        g = self.features.raw_functions(x0, side="right")
        mode = "aligned" if self.name == "upstream_cca" else "raw"
        out = np.column_stack([self.estimator.predict(f, g, h, mode=mode, basis="f")[:, 0]
                               for h in range(1, len(schedule)+1)])
        if not np.isfinite(out).all():
            raise FloatingPointError("nonfinite upstream forecast")
        return np.clip(out, -1., 1.)


def fit_official(train_obs, train_damping, val_obs, val_damping, *, seed=0,
                 matched_rank=4, steps=64, width=16):
    """Author encoders, loss, native inference; third head is our schedule extension."""
    if not os.environ.get("KOOPMAN_UPSTREAM"):
        raise RuntimeError("Set KOOPMAN_UPSTREAM to the pinned checkout (docs/pendulum-upstream.md)")
    module_cls, loss_cls, constant_cls, mlp_cls, estimator_cls = _upstream(os.environ["KOOPMAN_UPSTREAM"])
    train, damping = _check(train_obs, train_damping)
    val, _ = _check(val_obs, val_damping)
    if matched_rank < 2 or width < 1 or steps < 0 or val.shape[2:] != train.shape[2:]:
        raise ValueError("invalid rank, width, step budget or validation shape")
    t0 = time.perf_counter()
    d = train.shape[-1]
    def pairs(a):
        return [torch.as_tensor(b.reshape(-1, d), dtype=torch.float64)
                for b in (a[:, :, :-1], a[:, :, 1:])]
    x, y = pairs(train)
    xv, yv = pairs(val)
    xm, ym = x.mean(0), y.mean(0)
    xs, ys = x.std(0, unbiased=False), y.std(0, unbiased=False)
    xs = torch.where(xs > 1e-12, xs, 1.)
    ys = torch.where(ys > 1e-12, ys, 1.)
    xn, yn, xvn, yvn = (x-xm)/xs, (y-ym)/ys, (xv-xm)/xs, (yv-ym)/ys
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        def encoder():
            return constant_cls(mlp_cls(d, matched_rank-1, [width], activation="Tanh", use_batchnorm=False))
        f, g = encoder(), encoder()
        loss = loss_cls(matched_rank, nesting="jnt", reg_weight=0.)
        m = module_cls(f, loss, optimizer_kwargs={"lr": .003}, lagged_encoder=g,
                       learn_svals=False, num_svals=matched_rank, has_centering=True).double()
    def score(xi, yi):
        return m.loss_fn(m(xi), m(yi, lagged=True), m.svals)
    optimizer = m.configure_optimizers()
    gen = torch.Generator().manual_seed(int(seed)+104729)
    def checkpoint():
        return tuple({k: v.detach().clone() for k, v in net.state_dict().items()} for net in (f,g))
    with torch.no_grad():
        best = float(score(xvn, yvn))
    best_params, selected = checkpoint(), 0
    for step in range(1, steps+1):
        idx = torch.randint(len(xn), (128,), generator=gen)
        optimizer.zero_grad()
        # Authors' single-device NestedLoRALoss uses the *same paired batch*
        # for alignment and both Gram moments. Do not replace it with our loss.
        obj = score(xn[idx], yn[idx])
        if not torch.isfinite(obj):
            raise FloatingPointError("nonfinite upstream nested objective")
        obj.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        if step % 8 == 0 or step == steps:
            with torch.no_grad():
                val_score = float(score(xvn, yvn))
            if val_score < best:
                best, selected, best_params = val_score, step, checkpoint()
    for net, state in zip((f, g), best_params):
        net.load_state_dict(state)
        net.eval()
    features = _Features(m, xm, xs, ym, ys, matched_rank)
    stats = estimator_cls(rank=matched_rank, use_cca=True)
    xnp, ynp = x.numpy(), y.numpy()
    stats.partial_fit(features.raw_functions(xnp, side="left"), features.raw_functions(xnp, side="right"),
                      features.raw_functions(ynp, side="left"), features.raw_functions(ynp, side="right"),
                      x_raw=xnp[:, :1], y_raw=ynp[:, :1]).finalize()
    fit_seconds = time.perf_counter()-t0
    count = sum(p.numel() for p in m.parameters())
    meta = dict(dictionary_width=matched_rank, state_size=matched_rank, stored_network_parameters=count,
                trainable_parameters=count, model_bytes=8*(count+4*d), selected_step=selected, steps=steps,
                fit_seconds=fit_seconds, upstream_revision=REVISION, validation_nested_loss=best)
    out = {name: _NativeForecast(features, stats, name, meta) for name in ("upstream_cca","upstream_edmd")}
    tic = time.perf_counter()
    prefix = PrefixSpectralForecast(train, damping, matched_rank, features, "upstream_prefix")
    prefix.fit_seconds = fit_seconds + (time.perf_counter()-tic)
    prefix.metadata.update(trainable_parameters=count, selected_step=selected, steps=steps,
                           fit_seconds=prefix.fit_seconds, upstream_revision=REVISION,
                           validation_nested_loss=best, mode="our prefix-head adapter")
    out["upstream_prefix"] = prefix
    return out
