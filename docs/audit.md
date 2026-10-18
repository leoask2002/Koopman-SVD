# Repository audit and modernization scope

Base reviewed: `fb55ed603177f224f7b5aa5abc61c1c9e0ec8681` on `main` (last original push 2024-10-29). The accessible repository had three commits, a single main branch, and no pull requests at inspection. The later locally discussed logistic scripts were not present in this Git history; their contents have not been recovered or represented as audited.

## Findings

1. `koopman_operator_1.py` and `methods/koopman_operator.py` apply a fixed matrix to states. They neither identify dynamics nor implement general Koopman action on learned observables.
2. The added `KoopmanOperator` branch in `methods/nestedlora.py::compute_loss_operator` uses an unbound `f`, while `Tf` is a transformed state, not the transformed observable. Moreover, the EVD custom gradient assumes a self-adjoint operator, which a general Koopman operator is not. The unsupported legacy shortcut now fails explicitly and points to the new paired-data API.
3. There was no tested paired-data neural Koopman estimator, no train-only normalization/calibration, no proper singular-basis multi-step prediction, no installable package metadata, and no automated regression suite.
4. The upstream paper examples include historical optional dependencies and CUDA AMP decorators. They are preserved as reference experiments, not claimed to have been fully modernized or rerun. The new code is independent of those dependency chains.
5. Direct NeuralSVD/Koopman prior art appeared in NeurIPS 2025. Attribution and mathematical conventions are updated accordingly.

## Delivered path

The supported path runs from arrays or trajectory contexts through paired transition extraction, local-seed neural optimization with independent marginal batches, validation checkpoint selection, train-only post-fit calibration, diagnostics, conditional-mean forecasts, and checkpoint round trips. The benchmark CLI records exact configurations, versions, per-seed metrics, wall time, and validation stopping metadata. CI executes correctness tests and a bounded quick experiment on CPU.

Tests target the risk-bearing behavior: prefix objective and gradients; pairing and no boundary crossings; non-symmetric orientation; left/right overlap composition; known finite reference operators; train-only calibration; checkpoint and invalid input behavior. Full paper reproduction, long training, large-scale hyperparameter search and continuous-time/control synthesis remain outside the work performed.
