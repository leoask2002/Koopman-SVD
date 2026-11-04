"""Nondimensional noisy pendulum: controlled numerical benchmark, not hardware data.

A macro interval lasts DT, with constant damping gamma over its eight substeps.
The semi-implicit Euler–Maruyama update below *defines* the discrete benchmark.
Its continuous-time reference is dtheta = omega dt, domega = (-sin(theta) -
gamma omega) dt + SIGMA dW. Angles live on the circle; keeping an unwrapped
representative avoids branch discontinuities in integration. Evaluation starts
are at time zero. The initial latent angular velocity is independent of angle.
"""
from numbers import Integral

import numpy as np

DT = 0.25
SIGMA = 0.36
OMEGA_SD = 1.20
SUBSTEPS = 8
SCHEDULES = ("train_low", "train_high", "train_ramp", "train_step",
             "val_smooth", "val_switch", "test_smooth", "test_switch")


def _positive_int(x, name):
    if isinstance(x, bool) or not isinstance(x, Integral) or x < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(x)


def damping(name, steps=8):
    """Return deterministic damping values on successive macro intervals.

    The two test laws have a sinusoidal profile and a low-high-low switch.
    Neither profile is a fitted or validation law. All coefficients are within
    the broad training range, so this tests new schedules, not extrapolation to
    negative or unbounded damping.
    """
    k = _positive_int(steps, "steps")
    u = (np.arange(k) + .5) / k
    if name == "train_low":
        return np.full(k, .15)
    if name == "train_high":
        return np.full(k, 1.55)
    if name == "train_ramp":
        return .18 + 1.33 * u
    if name == "train_step":
        return np.where(u < .5, .24, 1.33)
    if name == "val_smooth":
        return 1.45 - 1.19 * u
    if name == "val_switch":
        return np.where(u < .62, 1.24, .32)
    if name == "test_smooth":
        return .83 + .60 * np.sin(2 * np.pi * (u + .13))
    if name == "test_switch":
        return np.where((u >= .23) & (u <= .70), 1.47, .19)
    raise ValueError(f"unknown damping schedule {name!r}; expected one of {SCHEDULES}")


def draw_starts(rng, n):
    n = _positive_int(n, "n")
    return np.column_stack((rng.uniform(-np.pi, np.pi, n),
                            rng.normal(0., OMEGA_SD, n)))


def _initial(starts, gammas):
    x = np.asarray(starts, dtype=float)
    g = np.asarray(gammas, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("starts must be a finite real (n,2) array")
    if g.ndim != 1 or not g.size or not np.isfinite(g).all() or (g < 0).any():
        raise ValueError("gammas must be a nonempty 1D nonnegative finite array")
    return x, g


def _update(theta, omega, gamma, dt, sigma_dw):
    omega = omega + (-np.sin(theta) - gamma * omega) * dt + sigma_dw
    return theta + dt * omega, omega


def _trajectory(theta, omega, gammas, rng, substeps, sigma):
    substeps = _positive_int(substeps, "substeps")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    dt = DT / substeps
    scale = sigma * np.sqrt(dt)
    for gamma in gammas:
        for _ in range(substeps):
            noise = scale * rng.standard_normal(theta.shape) if scale else 0.
            theta, omega = _update(theta, omega, gamma, dt, noise)
        yield theta, omega


def simulate(rng, starts, gammas, *, substeps=SUBSTEPS, sigma=SIGMA):
    """Simulate independent paths with latent shape (n, time+1, 2)."""
    x, g = _initial(starts, gammas)
    out = np.empty((len(x), len(g)+1, 2))
    out[:, 0] = x
    for t, (q, w) in enumerate(_trajectory(x[:, 0], x[:, 1], g, rng, substeps, sigma), 1):
        out[:, t, 0], out[:, t, 1] = q, w
    return out


# Two fixed observation maps. Both expose the same invariant anchors. Their
# nuisance channels deliberately have different meanings, not matched scale.
_MIX = {
    "A": np.array([[1.1, .6, -.3], [-.4, 1.4, .6], [.8, -.2, 1.5],
                   [-1., .5, .7], [.3, .9, -.8]]),
    "B": np.array([[-.7, .3, 1.3], [1.2, -.8, .4], [-.2, 1.4, -.9],
                   [.2, -.6, 1.7], [1.0, .7, .3]]),
}


def observe(latent, view="full", sensor="A"):
    """Eight bounded sensor channels, including the bounded target at index 0.

    The first three channels are cos(theta), sin(theta), and tanh(omega/2)
    (full) or sin(2*theta) (partial). The informative full-state map is
    injective in exact real arithmetic on S^1 x R. Partial measurements contain
    no angular velocity. The other channels are redundant nonlinear sensors.
    """
    x = np.asarray(latent, dtype=float)
    if x.ndim < 1 or x.shape[-1] != 2 or not np.isfinite(x).all():
        raise ValueError("latent must be a finite real array with final dimension 2")
    if view not in ("full", "partial") or sensor not in _MIX:
        raise ValueError("view must be full/partial and sensor must be A/B")
    q = x[..., 0]
    anchors = np.stack((np.cos(q), np.sin(q),
                        np.tanh(x[..., 1]/2) if view == "full" else np.sin(2*q)), axis=-1)
    extra = np.tanh(anchors @ _MIX[sensor].T + np.array([.2, -.1, .3, -.2, 0.]))
    return np.concatenate([anchors, extra], axis=-1)


def conditional_endpoints(rng, starts, gammas, *, view="full", branches=1):
    """Return endpoint cosines, shape (n, branches, time), all given time-zero information.

    Each partial branch resamples the independent initial velocity; conditioning
    on the full latent state in this case would leak unavailable information.
    Use independent calls/RNG streams for evaluation references and calibration.
    """
    x, g = _initial(starts, gammas)
    branches = _positive_int(branches, "branches")
    if view not in ("full", "partial"):
        raise ValueError("view must be full/partial")
    theta = np.broadcast_to(x[:, :1], (len(x), branches))
    omega = (np.broadcast_to(x[:, 1:], theta.shape) if view == "full"
             else rng.normal(0., OMEGA_SD, size=theta.shape))
    out = np.empty((*theta.shape, len(g)))
    for t, (theta, _) in enumerate(_trajectory(theta, omega, g, rng, SUBSTEPS, SIGMA)):
        out[:, :, t] = np.cos(theta)
    return out


def reference_scores(pred, ref):
    """Unbiased conditional-mean MSE estimate; RMSE is a descriptive transform.

    pred is (..., n, h) and ref is (n, R, h), R>=2 independent replicates per
    start. The sample mean has conditional variance s_i²/R in expectation;
    subtract it before averaging over starts. Retain signed MSE and independent
    start standard errors; only the reported square root is clamped at zero.
    """
    pred, ref = np.asarray(pred, dtype=float), np.asarray(ref, dtype=float)
    if (ref.ndim != 3 or ref.shape[1] < 2 or ref.shape[0] < 2 or pred.ndim < 2
            or pred.shape[-2:] != (ref.shape[0], ref.shape[2])):
        raise ValueError("expected (..., n, h) forecasts and (n, R, h) references, n,R >= 2")
    if not np.isfinite(ref).all() or not np.isfinite(pred).all():
        raise ValueError("pred and ref must be finite")
    correction = ref.var(axis=1, ddof=1) / ref.shape[1]
    scores = (pred - ref.mean(axis=1))**2
    z = scores - correction
    mse = z.mean(axis=-2)
    return dict(raw_mse=scores.mean(axis=-2), mse=mse, rmse=np.sqrt(np.maximum(0, mse)),
                reference_noise_floor=correction.mean(axis=0),
                mse_se=z.std(axis=-2, ddof=1) / np.sqrt(ref.shape[0]))


def physics_check(seed=24017, n=1024):
    """Coupled coarse/fine comparisons plus a noise-free high-order reference.

    Finite-sample diagnostics of this benchmark, not a bound for every path.
    The coarse and fine SDE updates sum identical Brownian increments.
    """
    rng = np.random.default_rng(seed)
    starts = draw_starts(rng, n)
    g = damping("test_switch", 8)
    multiples = (1, 2, 4)
    states = {k: starts.copy() for k in multiples}
    for gamma in g:
        dw = SIGMA * np.sqrt(DT / (4*SUBSTEPS)) * rng.standard_normal((4*SUBSTEPS, n))
        for k in multiples:
            theta, omega = states[k].T
            m = SUBSTEPS*k
            for delta in dw.reshape(m, 4//k, n).sum(axis=1):
                theta, omega = _update(theta, omega, gamma, DT/m, delta)
            states[k] = np.column_stack((theta, omega))
    def cosine_diff(a, b):
        return float(np.sqrt(np.mean((np.cos(a[:, 0]) - np.cos(b[:, 0]))**2)))
    rk = starts.copy()
    def drift(x, gamma):
        return np.stack((x[:, 1], -np.sin(x[:, 0])-gamma*x[:, 1]), axis=1)
    for gamma in g:
        dt = DT / 128
        for _ in range(128):
            k1 = drift(rk, gamma)
            k2 = drift(rk + .5*dt*k1, gamma)
            k3 = drift(rk + .5*dt*k2, gamma)
            k4 = drift(rk + dt*k3, gamma)
            rk = rk + dt/6*(k1+2*k2+2*k3+k4)
    deterministic = {}
    for k in multiples:
        paths = simulate(rng, starts, g, sigma=0., substeps=k*SUBSTEPS)
        deterministic[str(k*SUBSTEPS)] = cosine_diff(paths[:, -1], rk)

    return {"seed":seed, "sampled_starts":n, "schedule":"test_switch", "horizon":8,
            "dt_macro":DT, "substeps":SUBSTEPS, "sigma":SIGMA,
            "sde_coupled_endpoint_cosine_rmse":{
                "8_vs_16":cosine_diff(states[1], states[2]),
                "16_vs_32":cosine_diff(states[2], states[4]),
                "8_vs_32":cosine_diff(states[1], states[4])},
            "ode_endpoint_cosine_rmse_vs_RK4_128":deterministic}


if __name__ == "__main__":
    import json
    print(json.dumps(physics_check(), indent=2))
