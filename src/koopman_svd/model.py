"""Neural singular functions and a calibrated finite-rank Koopman operator.

The orientation throughout is ``K h(x) = E[h(Y) | X=x]``.  Source functions
are left singular functions and target functions are right singular functions.
Optimization uses the NeuralSVD joint-nesting moment objective with ordinary
autodiff.  A separate, train-only, small matrix calibration orthonormalizes the
learned feature spaces and fits a state decoder.  Consequently calibrated
spectra describe the learned *spaces*, and do not alone establish convergence
of the neural singular functions themselves.
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .losses import nesting_masks, neural_svd_loss
from .networks import MLPFeatures


class NeuralKoopmanSVD:
    """Learn a bounded-rank approximation from paired state observations.

    Parameters
    ----------
    input_dim : int
        Shared dimension of source and target states.
    rank : int
        Total number of functions, including a fixed constant when enabled.
    hidden_dims : sequence of int
        Empty means affine features: calibration is then an affine EDMD
        sanity check, not evidence of a nonlinear neural advantage.
    feature_map : callable, optional
        Fixed tensor-to-tensor map applied after train-only standardization.
        The callable must accept a two-dimensional tensor and preserve rows.
        It is not serialized; pass it again when loading a saved estimator.
    include_constant : bool
        Reserve one output for the exact constant function.  It preserves
        affine offsets under calibration and projected operator composition.
    calibration_rcond : float
        Relative Gram eigenvalue threshold.  Unsupported empirical directions
        are discarded and reported rather than returning spurious modes.

    Notes
    -----
    The model assumes one common state domain, including when the empirical
    source and target distributions differ.  Multi-step predictions compose
    the fitted finite-rank kernel using a same-state overlap under the target
    marginal.  They require evaluating source functions on target states and
    are projected continuations, not a guarantee of out-of-distribution
    dynamics or global Koopman singular values.
    """

    def __init__(
        self,
        input_dim: int,
        rank: int,
        hidden_dims=(32, 32),
        *,
        seed: int = 0,
        dtype=torch.float64,
        device="cpu",
        feature_map=None,
        include_constant: bool = True,
        calibration_rcond: float = 1e-10,
    ):
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim < 1:
            raise ValueError("input_dim must be a positive integer")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("rank must be a positive integer")
        if any(isinstance(h, bool) or not isinstance(h, int) or h < 1 for h in hidden_dims):
            raise ValueError("hidden_dims must contain positive integers")
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype.removeprefix("torch."), None)
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        if not 0 < calibration_rcond < 1:
            raise ValueError("calibration_rcond must be between zero and one")
        self.input_dim = input_dim
        self.rank = rank
        self.hidden_dims = tuple(hidden_dims)
        self.seed = int(seed)
        self.dtype = dtype
        self.device = torch.device(device)
        self.feature_map = feature_map
        self.include_constant = bool(include_constant)
        self.calibration_rcond = float(calibration_rcond)
        self._trained = False
        self._calibrated = False
        self.history_ = []
        self.fit_summary_ = {}
        self.calibration_summary_ = {}
        self._initialize_networks()

    def _map_features(self, x: Tensor) -> Tensor:
        mapped = x if self.feature_map is None else self.feature_map(x)
        if not isinstance(mapped, Tensor) or mapped.ndim != 2 or mapped.shape[0] != x.shape[0]:
            raise ValueError("feature_map must return a 2D tensor preserving the batch size")
        if mapped.shape[1] < 1 or not bool(torch.isfinite(mapped).all()):
            raise ValueError("feature_map must return finite, nonempty features")
        if mapped.dtype != self.dtype or mapped.device != self.device:
            raise ValueError("feature_map must preserve dtype and device")
        return mapped

    def _initialize_networks(self):
        with torch.no_grad():
            feature_dim = self._map_features(torch.zeros((2, self.input_dim), dtype=self.dtype, device=self.device)).shape[1]
        # fork_rng restores global RNG state; this estimator does not seed the
        # user's NumPy or PyTorch session as a side effect.
        # Layers initialize on CPU before moving to the requested device, so
        # only the CPU generator needs a local seed (including on GPU runs).
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            output_dim = self.rank - int(self.include_constant)
            self.source_net_ = MLPFeatures(feature_dim, output_dim, self.hidden_dims).to(device=self.device, dtype=self.dtype)
            self.target_net_ = MLPFeatures(feature_dim, output_dim, self.hidden_dims).to(device=self.device, dtype=self.dtype)

    def _array(self, value, name="states") -> Tensor:
        # Complex Koopman observables require conjugate Gram products and a
        # different network contract; silently discarding their imaginary
        # parts would identify an unintended operator.
        if (isinstance(value, Tensor) and value.is_complex()) or (
            not isinstance(value, Tensor) and np.iscomplexobj(value)
        ):
            raise ValueError(f"{name} must be real-valued; complex states are not supported")
        try:
            value = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"{name} must be a finite numeric matrix") from exc
        if value.ndim != 2 or value.shape[1] != self.input_dim or value.shape[0] < 1:
            raise ValueError(f"{name} must have shape (n, {self.input_dim}) with n >= 1")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains nonfinite values")
        return value.detach()

    def _paired_arrays(self, x, y):
        x, y = self._array(x, "x"), self._array(y, "y")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must contain equal numbers of paired states")
        return x, y

    def _raw(self, x: Tensor, side: str) -> Tensor:
        if side == "left":
            normalized = (x - self.x_mean_) / self.x_scale_
            learned = self.source_net_(self._map_features(normalized))
        elif side == "right":
            normalized = (x - self.y_mean_) / self.y_scale_
            learned = self.target_net_(self._map_features(normalized))
        else:
            raise ValueError("side must be 'left' or 'right'")
        if self.include_constant:
            return torch.cat((x.new_ones((len(x), 1)), learned), dim=1)
        return learned

    def _state_dict(self):
        return {"source": copy.deepcopy(self.source_net_.state_dict()), "target": copy.deepcopy(self.target_net_.state_dict())}

    def _restore(self, state):
        self.source_net_.load_state_dict(state["source"])
        self.target_net_.load_state_dict(state["target"])

    @torch.no_grad()
    def _objective(self, x, y, chunk_size=4096):
        # Exact empirical moments, accumulated in chunks to avoid retaining an
        # evaluation autograd graph or a full feature matrix.
        c_f = x.new_zeros((self.rank, self.rank))
        c_g = x.new_zeros((self.rank, self.rank))
        cross = x.new_zeros(self.rank)
        for start in range(0, len(x), chunk_size):
            f, g = self._raw(x[start:start + chunk_size], "left"), self._raw(y[start:start + chunk_size], "right")
            c_f += f.T @ f
            c_g += g.T @ g
            cross += (f * g).sum(dim=0)
        a, w = nesting_masks(self.rank, dtype=self.dtype, device=self.device)
        return float(-2 * torch.sum(a * cross / len(x)) + torch.sum(w * c_f * c_g / len(x) ** 2))

    def fit(
        self,
        x,
        y,
        *,
        validation_data=None,
        steps: int = 300,
        batch_size: int = 256,
        lr: float = 0.003,
        patience: int | None = 20,
        validation_every: int = 10,
        max_seconds: float | None = 60.0,
        min_delta: float = 1e-7,
        gradient_clip: float = 10.0,
    ):
        """Fit neural functions, then calibrate using only the training pairs.

        Validation controls checkpoint selection but never normalization,
        whitening, cross-SVD, the decoder, or the continuation matrix.
        ``patience`` counts validation evaluations without improvement.
        ``steps=0`` is an explicitly untrained random-feature baseline.
        ``max_seconds`` bounds the optimization loop (not input validation or
        the final small calibration); no background training is launched.
        """
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(validation_every, int) or validation_every < 1:
            raise ValueError("validation_every must be a positive integer")
        if patience is not None and (not isinstance(patience, int) or patience < 1):
            raise ValueError("patience must be a positive integer or None")
        if not math.isfinite(lr) or lr <= 0 or not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("lr must be positive and min_delta nonnegative")
        if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
            raise ValueError("max_seconds must be positive or None")
        if not math.isfinite(gradient_clip) or gradient_clip <= 0:
            raise ValueError("gradient_clip must be finite and positive")
        x, y = self._paired_arrays(x, y)
        if len(x) < self.rank:
            raise ValueError("at least rank training pairs are needed for calibration")
        if validation_data is None:
            xv, yv = x, y
        else:
            if not isinstance(validation_data, (tuple, list)) or len(validation_data) != 2:
                raise ValueError("validation_data must be an (x_validation, y_validation) pair")
            xv, yv = self._paired_arrays(*validation_data)
        self._calibrated = False
        self._initialize_networks()
        self.x_mean_, self.y_mean_ = x.mean(0), y.mean(0)
        # Constant state coordinates keep unit scale rather than amplify noise.
        self.x_scale_, self.y_scale_ = x.std(0, unbiased=False), y.std(0, unbiased=False)
        scale_floor = math.sqrt(torch.finfo(self.dtype).eps)
        self.x_scale_ = torch.where(self.x_scale_ > scale_floor, self.x_scale_, torch.ones_like(self.x_scale_))
        self.y_scale_ = torch.where(self.y_scale_ > scale_floor, self.y_scale_, torch.ones_like(self.y_scale_))
        self._trained = True
        parameters = list(self.source_net_.parameters()) + list(self.target_net_.parameters())
        optimizer = torch.optim.Adam(parameters, lr=lr) if parameters else None
        generator = torch.Generator(device=self.device).manual_seed(self.seed + 104729)
        started = time.perf_counter()
        best_loss = self._objective(xv, yv)
        if not math.isfinite(best_loss):
            raise ValueError("initial objective is nonfinite; check feature_map and data scaling")
        best_state = self._state_dict()
        self.history_ = [{"step": 0, "validation_loss": best_loss, "train_loss": self._objective(x, y)}]
        best_step, completed, stale = 0, 0, 0
        reason = "constant_only" if not parameters else "step_limit"
        for step in range(1, steps + 1) if parameters else ():
            if max_seconds is not None and time.perf_counter() - started >= max_seconds:
                reason = "time_limit"
                break
            # Three independent draws: paired observations, source marginal,
            # target marginal.  Equal batch sizes do not imply shared indices.
            paired_idx = torch.randint(len(x), (batch_size,), generator=generator, device=self.device)
            source_idx = torch.randint(len(x), (batch_size,), generator=generator, device=self.device)
            target_idx = torch.randint(len(y), (batch_size,), generator=generator, device=self.device)
            optimizer.zero_grad(set_to_none=True)
            loss = neural_svd_loss(self._raw(x[paired_idx], "left"), self._raw(y[paired_idx], "right"), self._raw(x[source_idx], "left"), self._raw(y[target_idx], "right"))
            if not bool(torch.isfinite(loss)):
                reason = "nonfinite_loss"
                break
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)
            if not bool(torch.isfinite(grad_norm)):
                reason = "nonfinite_gradient"
                break
            optimizer.step()
            completed = step
            if step % validation_every == 0 or step == steps:
                score = self._objective(xv, yv)
                self.history_.append({"step": step, "validation_loss": score, "train_loss": self._objective(x, y), "minibatch_loss": float(loss.detach())})
                if not math.isfinite(score):
                    reason = "nonfinite_validation"
                    break
                if score < best_loss - min_delta:
                    best_loss, best_step, best_state, stale = score, step, self._state_dict(), 0
                else:
                    stale += 1
                if patience is not None and stale >= patience:
                    reason = "early_stopping"
                    break
        self._restore(best_state)
        self.fit_summary_ = {"steps_completed": completed, "best_step": best_step, "best_validation_loss": best_loss, "stopping_reason": reason, "optimization_seconds": time.perf_counter() - started, "training_pairs": len(x), "validation_pairs": len(xv) if validation_data is not None else 0, "selection_data": "validation" if validation_data is not None else "training", "seed": self.seed, "trained": best_step > 0, "optimization_performed": completed > 0}
        return self.calibrate(x, y)

    def _whitener(self, gram, label):
        # A fixed constant is kept exact.  The means were subtracted only from
        # nonconstant coordinates, so their empirical cross moments are zero.
        offset = int(self.include_constant)
        block = gram[offset:, offset:]
        if block.numel() == 0:
            return torch.ones_like(gram), 1, 1.0
        eigenvalues, vectors = torch.linalg.eigh((block + block.T) / 2)
        largest = float(eigenvalues.max())
        # A user tolerance below roundoff cannot recover genuine directions;
        # in float32 especially, retain only numerically resolvable modes.
        relative_cutoff = max(self.calibration_rcond, torch.finfo(self.dtype).eps * self.rank)
        cutoff = max(largest * relative_cutoff, torch.finfo(self.dtype).tiny)
        keep = eigenvalues > cutoff
        effective = int(keep.sum()) + offset
        if not effective:
            raise ValueError(f"{label} feature Gram matrix has zero effective rank; increase feature expressiveness")
        whitening = gram.new_zeros((self.rank, effective))
        if offset:
            whitening[0, 0] = 1.0
        whitening[offset:, offset:] = vectors[:, keep] * eigenvalues[keep].rsqrt()
        condition = largest / float(eigenvalues[keep].min()) if bool(keep.any()) else 1.0
        return whitening, effective, condition

    @torch.no_grad()
    def calibrate(self, x, y):
        """Calibrate small matrices from training pairs only; never use test data.

        This explicit postfit stage uses Gram eigendecompositions and a cross
        SVD.  These factorizations are not part of neural optimization.
        Calling this method again deliberately replaces the calibration data.
        """
        if not self._trained:
            raise RuntimeError("fit must establish training normalization before calibrate")
        self._calibrated = False
        x, y = self._paired_arrays(x, y)
        if len(x) < self.rank:
            raise ValueError("at least rank pairs are needed for calibration")
        f_raw, g_raw = self._raw(x, "left"), self._raw(y, "right")
        raw_cf, raw_cg = f_raw.T @ f_raw / len(x), g_raw.T @ g_raw / len(x)
        self.raw_singular_values_ = torch.sqrt(torch.diag(raw_cf) * torch.diag(raw_cg)).cpu().numpy()
        self.raw_cross_diagonal_ = (f_raw * g_raw).mean(0).cpu().numpy()
        self._raw_cf, self._raw_cg = raw_cf, raw_cg
        self.f_center_ = f_raw.mean(0) if self.include_constant else torch.zeros(self.rank, dtype=self.dtype, device=self.device)
        self.g_center_ = g_raw.mean(0) if self.include_constant else torch.zeros(self.rank, dtype=self.dtype, device=self.device)
        if self.include_constant:
            self.f_center_[0] = self.g_center_[0] = 0.0
        f, g = f_raw - self.f_center_, g_raw - self.g_center_
        cf, cg, cross = f.T @ f / len(x), g.T @ g / len(x), f.T @ g / len(x)
        self.calibration_summary_ = {"pairs": len(x), "rank": self.rank, "includes_constant": self.include_constant}
        wf, rank_f, cond_f = self._whitener(cf, "source")
        wg, rank_g, cond_g = self._whitener(cg, "target")
        u, singular, vh = torch.linalg.svd(wf.T @ cross @ wg, full_matrices=False)
        self._left_transform, self._right_transform = wf @ u, wg @ vh.T
        self.effective_rank_ = len(singular)
        self._singular_values = singular
        self.singular_values_ = singular.cpu().numpy()
        phi, psi = f @ self._left_transform, g @ self._right_transform
        self._observable_coefficients = psi.T @ y / len(y)
        # SAME-state overlap, not phi(X) paired with psi(Y).  This is crucial:
        # singular values alone cannot be raised to powers for nonnormal K.
        phi_on_y = (self._raw(y, "left") - self.f_center_) @ self._left_transform
        self._bridge = psi.T @ phi_on_y / len(y)
        self._continuation = singular[:, None] * self._bridge
        self._decoder = singular[:, None] * self._observable_coefficients
        identity = torch.eye(self.effective_rank_, dtype=self.dtype, device=self.device)
        self.calibration_summary_.update({"requested_rank": self.rank, "retained_rank": self.effective_rank_, "rank_truncated": self.effective_rank_ < self.rank, "source_effective_rank": rank_f, "target_effective_rank": rank_g, "source_gram_condition": cond_f, "target_gram_condition": cond_g, "left_orthogonality_error": float(torch.linalg.norm(phi.T @ phi / len(x) - identity)), "right_orthogonality_error": float(torch.linalg.norm(psi.T @ psi / len(y) - identity)), "cross_diagonalization_error": float(torch.linalg.norm(phi.T @ psi / len(x) - torch.diag(singular))), "continuation_spectral_radius": float(torch.linalg.eigvals(self._continuation).abs().max()), "method": "train-only centered Gram whitening and cross-SVD"})
        self._calibrated = True
        return self

    def _require_fitted(self):
        if not self._calibrated:
            raise RuntimeError("fit must complete successfully before inference")

    @staticmethod
    def _return(tensor, original):
        tensor = tensor.detach()
        return tensor if isinstance(original, Tensor) else tensor.cpu().numpy()

    @torch.no_grad()
    def raw_functions(self, x, *, side="left"):
        """Return the uncalibrated learned factors, including any constant."""
        if not self._trained:
            raise RuntimeError("fit must run before evaluating raw functions")
        return self._return(self._raw(self._array(x), side), x)

    @torch.no_grad()
    def singular_functions(self, x, *, side="left"):
        self._require_fitted()
        raw = self._raw(self._array(x), side)
        center, transform = (self.f_center_, self._left_transform) if side == "left" else (self.g_center_, self._right_transform)
        return self._return((raw - center) @ transform, x)

    def left_functions(self, x):
        """Calibrated left/output singular functions on source states."""
        return self.singular_functions(x, side="left")

    def right_functions(self, y):
        """Calibrated right/input singular functions on target states."""
        return self.singular_functions(y, side="right")

    @torch.no_grad()
    def predict(self, x, *, steps=1):
        """Apply the fitted operator ``steps`` times to coordinate observables.

        For n>=1 this computes phi(x) (Sigma H)^(n-1) Sigma b.  It does not
        iterate a nonlinear predicted state, or treat singular values as
        eigenvalues.  At n=0 the input is returned exactly.
        """
        self._require_fitted()
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        values = self._array(x)
        if steps == 0:
            return self._return(values.clone(), x)
        phi = (self._raw(values, "left") - self.f_center_) @ self._left_transform
        propagated = self._decoder if steps == 1 else torch.linalg.matrix_power(self._continuation, steps - 1) @ self._decoder
        result = phi @ propagated
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("projected operator powers became nonfinite")
        return self._return(result, x)

    @torch.no_grad()
    def rollout(self, x, horizon):
        """Return shape (samples, horizon+1, state_dim), including initial states."""
        self._require_fitted()
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0:
            raise ValueError("horizon must be a nonnegative integer")
        values = self._array(x)
        phi = (self._raw(values, "left") - self.f_center_) @ self._left_transform
        outputs = [values]
        coefficients = self._decoder
        for _ in range(horizon):
            outputs.append(phi @ coefficients)
            coefficients = self._continuation @ coefficients
        result = torch.stack(outputs, dim=1)
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("projected operator powers became nonfinite")
        return self._return(result, x)

    @torch.no_grad()
    def diagnostics(self, x, y):
        """Read-only held-out diagnostics; does not recalibrate any quantity.

        ``prediction_rmse`` includes irreducible noise for stochastic systems.
        A small training calibration error is algebraic, not evidence of
        held-out accuracy.  Held-out orthogonality may differ with sampling or
        a changed marginal distribution.
        """
        self._require_fitted()
        x, y = self._paired_arrays(x, y)
        phi, psi = self.left_functions(x), self.right_functions(y)
        f, g = self._raw(x, "left"), self._raw(y, "right")
        prediction = self.predict(x)
        residual = prediction - y
        mse = (residual.square()).mean()
        variance = (y - y.mean(0)).square().mean()
        cf, cg = f.T @ f / len(x), g.T @ g / len(x)
        identity = torch.eye(self.effective_rank_, dtype=self.dtype, device=self.device)
        raw_identity = torch.eye(self.rank, dtype=self.dtype, device=self.device)
        norm_f = torch.diag(cf).clamp_min(torch.finfo(self.dtype).tiny).sqrt()
        norm_g = torch.diag(cg).clamp_min(torch.finfo(self.dtype).tiny).sqrt()
        normalized_cf, normalized_cg = cf / (norm_f[:, None] * norm_f[None, :]), cg / (norm_g[:, None] * norm_g[None, :])
        return {"n_pairs": len(x), "prediction_rmse": float(mse.sqrt()), "prediction_relative_rmse": float((mse / variance).sqrt()) if float(variance) > 0 else None, "r2": float(1 - mse / variance) if float(variance) > 0 else None, "neural_objective": self._objective(x, y), "singular_values": self.singular_values_.tolist(), "raw_norm_products": torch.sqrt(torch.diag(cf) * torch.diag(cg)).cpu().tolist(), "raw_cross_diagonal": (f * g).mean(0).cpu().tolist(), "raw_left_orthogonality_error": float(torch.linalg.norm(normalized_cf - raw_identity)), "raw_right_orthogonality_error": float(torch.linalg.norm(normalized_cg - raw_identity)), "left_orthogonality_error": float(torch.linalg.norm(phi.T @ phi / len(x) - identity)), "right_orthogonality_error": float(torch.linalg.norm(psi.T @ psi / len(y) - identity)), "cross_diagonalization_error": float(torch.linalg.norm(phi.T @ psi / len(x) - torch.diag(self._singular_values))), "calibration": dict(self.calibration_summary_), "fit": dict(self.fit_summary_)}

    def save(self, path):
        """Save inference weights and calibration; no executable pickle objects.

        A custom feature_map must be supplied separately to ``load``.  Files
        are loaded with PyTorch's restricted ``weights_only=True`` mechanism.
        """
        self._require_fitted()
        tensors = ("x_mean_", "y_mean_", "x_scale_", "y_scale_", "f_center_", "g_center_", "_left_transform", "_right_transform", "_singular_values", "_observable_coefficients", "_bridge", "_continuation", "_decoder", "_raw_cf", "_raw_cg")
        payload = {"format_version": 1, "config": {"input_dim": self.input_dim, "rank": self.rank, "hidden_dims": list(self.hidden_dims), "seed": self.seed, "dtype": str(self.dtype).removeprefix("torch."), "include_constant": self.include_constant, "calibration_rcond": self.calibration_rcond}, "requires_feature_map": self.feature_map is not None, "networks": self._state_dict(), "tensors": {key: getattr(self, key).detach().cpu() for key in tensors}, "raw_singular_values": self.raw_singular_values_.tolist(), "raw_cross_diagonal": self.raw_cross_diagonal_.tolist(), "history": self.history_, "fit_summary": self.fit_summary_, "calibration_summary": self.calibration_summary_}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path, *, device="cpu", feature_map=None):
        """Restore a saved estimator on the requested device."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1:
            raise ValueError("unsupported NeuralKoopmanSVD checkpoint version")
        if payload["requires_feature_map"] and feature_map is None:
            raise ValueError("this checkpoint requires the original fixed feature_map")
        if not payload["requires_feature_map"] and feature_map is not None:
            raise ValueError("this checkpoint was fitted without a feature_map")
        model = cls(**payload["config"], device=device, feature_map=feature_map)
        model._restore(payload["networks"])
        for key, value in payload["tensors"].items():
            setattr(model, key, value.to(device=model.device, dtype=model.dtype))
        model.singular_values_ = model._singular_values.cpu().numpy()
        model.effective_rank_ = len(model._singular_values)
        model.raw_singular_values_ = np.asarray(payload["raw_singular_values"])
        model.raw_cross_diagonal_ = np.asarray(payload["raw_cross_diagonal"])
        model.history_, model.fit_summary_, model.calibration_summary_ = payload["history"], payload["fit_summary"], payload["calibration_summary"]
        model._trained = model._calibrated = True
        return model
