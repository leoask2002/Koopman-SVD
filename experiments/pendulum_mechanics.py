"""Mechanics-informed reference, privileged by its equation and sensor calibration.

Fits unknown force, damping response and additive diffusion from observed *macro*
transitions, never from simulator coefficients, test states or test outcomes.
It applies only to full calibrated observations, not the angle-only view.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

try:
    from .pendulum_system import DT, SUBSTEPS
except ImportError:  # Standalone experiment scripts launched from the repository root.
    from pendulum_system import DT, SUBSTEPS


def _decode(x):
    x = np.asarray(x, dtype=float)
    if x.shape[-1] < 3 or not np.isfinite(x[..., :3]).all():
        raise ValueError("expected three finite calibrated full-state anchors")
    c, s, v = np.moveaxis(x[..., :3], -1, 0)
    if not np.allclose(c*c + s*s, 1., atol=1e-7) or np.any(np.abs(v) >= 1):
        raise ValueError("invalid calibrated full-state anchors")
    return np.arctan2(s, c), 2*np.arctanh(v)


def _deterministic(q, w, gamma, pars, sensitivity):
    """One macro step with optional exact endpoint sensitivities to (a,b)."""
    a, b = pars
    dt = DT/SUBSTEPS
    alpha = 1 - b * gamma * dt
    if sensitivity:
        dq = np.zeros((len(w), 2))
        dw = dq.copy()
    for _ in range(SUBSTEPS):
        s = np.sin(q)
        if sensitivity:
            dw = alpha[:, None]*dw - (dt*a*np.cos(q))[:, None]*dq
            dw[:, 0] -= dt*s
            dw[:, 1] -= dt*gamma*w
            dq = dq + dt*dw
        w = alpha*w - dt*a*s
        q = q + dt*w
    return (w, dw) if sensitivity else w


def _moment_step(mq, mw, vq, vw, qw, gamma, a, b, sigma):
    """Exact first two moments of one update for a Gaussian input law.

    We approximate its generally non-Gaussian output by a Gaussian with these
    moments before the next substep. Noise is added to velocity BEFORE the angle
    update, so both process and cross covariance propagate.
    """
    dt = DT/SUBSTEPS
    beta = dt*a
    alpha = 1 - dt*b*gamma
    atten = np.exp(-0.5*vq)
    es, ec = atten*np.sin(mq), atten*np.cos(mq)
    vs = np.maximum(0., 0.5*(1.-np.exp(-2.*vq)*np.cos(2.*mq)) - es*es)
    mw_next = alpha*mw - beta*es
    vw_next = np.maximum(0., alpha**2*vw + beta**2*vs - 2*alpha*beta*qw*ec
                         + dt*sigma**2)
    cov = alpha*qw - beta*vq*ec
    return (mq+dt*mw_next, mw_next,
            np.maximum(0., vq+2*dt*cov+dt**2*vw_next),
            vw_next, cov+dt*vw_next)


@dataclass
class MechanicsReference:
    restoring: float
    damping_response: float
    diffusion: float
    name = "mechanics reference"
    state_size = 5  # two means and three covariance entries
    dictionary_width = 0  # no EDMD dictionary

    @property
    def metadata(self):
        return dict(name=self.name, restoring=float(self.restoring),
                    damping_response=float(self.damping_response), diffusion=float(self.diffusion),
                    state_size=self.state_size, dictionary_width=self.dictionary_width,
                    fit_parameters=3, privileged="known mechanics + full sensor calibration",
                    propagation="Gaussian moment closure on unwrapped angle and velocity")

    def predict(self, x0, damping):
        x = np.asarray(x0)
        g = np.asarray(damping, dtype=float)
        if x.ndim != 2 or g.ndim != 1 or len(g) == 0 or not np.isfinite(g).all() or np.any(g < 0):
            raise ValueError("expected (starts,sensors) and a nonnegative damping schedule")
        mq, mw = _decode(x)
        vq = np.zeros_like(mq)
        vw = np.zeros_like(mq)
        qw = np.zeros_like(mq)
        values = []
        for gamma in g:
            for _ in range(SUBSTEPS):
                mq, mw, vq, vw, qw = _moment_step(mq, mw, vq, vw, qw, gamma,
                    self.restoring, self.damping_response, self.diffusion)
            values.append(np.exp(-vq/2)*np.cos(mq))
        return np.stack(values, axis=-1)


def fit_mechanics(train_obs, train_damping):
    """Estimate unknown coefficients from independent full-state training paths.

    Drift: constrained least squares of macro velocity against integrated noise-free
    dynamics. Diffusion: match the mean velocity innovation square to the
    approximate one-step conditional variance. Gamma values are observed inputs.
    This is a small-noise quasi-estimator, not an exact likelihood estimator.
    """
    x = np.asarray(train_obs)
    g = np.asarray(train_damping, dtype=float)
    if x.ndim != 4 or x.shape[2] < 2 or g.shape != (x.shape[0], x.shape[2]-1):
        raise ValueError("expected (schedules,paths,time+1,sensors) and (schedules,time)")
    if not np.isfinite(g).all() or np.any(g < 0):
        raise ValueError("damping schedules must be finite and nonnegative")
    q, w = _decode(x)
    q0, w0, w1 = q[:, :, :-1].ravel(), w[:, :, :-1].ravel(), w[:, :, 1:].ravel()
    gamma = np.broadcast_to(g[:, None, :], w[:, :, :-1].shape).ravel()
    design = np.column_stack((-DT*np.sin(q0), -DT*gamma*w0))
    p, _, rank, _ = np.linalg.lstsq(design, w1-w0, rcond=None)
    if rank != 2:
        raise ValueError("training paths lack excitation to identify force and damping")
    p = np.maximum(p, 0.)
    for _ in range(6):
        wp, jac = _deterministic(q0, w0, gamma, p, True)
        step = np.linalg.lstsq(jac, w1-wp, rcond=None)[0]
        old = np.mean((w1-wp)**2)
        for scale in (1., .5, .25, .125):
            candidate = np.maximum(p + scale*step, 0.)
            pred = _deterministic(q0, w0, gamma, candidate, False)
            if np.mean((w1-pred)**2) <= old:
                p = candidate
                break
        if np.linalg.norm(step) < 1e-9:
            break

    # A unit diffusion first gives a scale; repeat moment matching to account
    # for weak variance dependence of the trigonometric Gaussian moments.
    sigma = 1.
    for _ in range(3):
        mq, mw = q0, w0
        zeros = np.zeros_like(w0)
        vq, vw, qw = zeros, zeros, zeros
        for _ in range(SUBSTEPS):
            mq, mw, vq, vw, qw = _moment_step(mq, mw, vq, vw, qw, gamma, *p, sigma)
        variance = np.mean(vw)
        if variance < 1e-24:
            break
        sigma *= np.sqrt(np.mean((w1-mw)**2) / variance)
    return MechanicsReference(*p, sigma)


def run_audit(out):
    """Fresh high-precision diagnostic, separate from primary model selection."""
    import json
    from pathlib import Path
    from pendulum_system import conditional_endpoints, damping, draw_starts, observe, reference_scores, simulate

    rng = lambda *ids: np.random.default_rng(np.random.SeedSequence(ids))
    gs = np.stack([damping(s, 6) for s in ("train_low", "train_high", "train_ramp", "train_step")])
    audits = []
    for seed in range(71, 76):
        tr = []
        for j, g in enumerate(gs):
            r = rng(seed, 10, j)
            tr.append(observe(simulate(r, draw_starts(r, 256), g)))
        fitted = fit_mechanics(np.stack(tr), gs)
        ignore_noise = MechanicsReference(fitted.restoring, fitted.damping_response, 0.)
        rec = {"seed": seed, "coefficients": fitted.metadata, "schedules": {}}
        for j, name in enumerate(("test_smooth", "test_switch")):
            g = damping(name, 6)
            starts = draw_starts(rng(seed, 60, j), 128)
            full = observe(starts)
            ref = conditional_endpoints(rng(seed, 61, j), starts, g, branches=1024)
            p = np.stack([fitted.predict(full, g), ignore_noise.predict(full, g)])
            score = reference_scores(p, ref)
            if not np.array_equal(p[0], fitted.predict(observe(starts, sensor="B"), g)):
                raise RuntimeError("invariant anchors changed across nuisance map")
            rec["schedules"][name] = {k: v.tolist() for k,v in score.items()}
        audits.append(rec)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seeds": list(range(71,76)), "models": ["moments","no_diffusion"],
                              "fresh_starts": 128, "branches": 1024, "fits": audits},
                              indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    import argparse
    cli = argparse.ArgumentParser(description=run_audit.__doc__)
    cli.add_argument("--output", default="reports/pendulum/mechanics.json")
    run_audit(cli.parse_args().output)
