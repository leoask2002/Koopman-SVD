# Mathematical conventions and validity limits

This package implements neural singular-function learning from time-lagged pairs. The neural networks are optimized with the jointly nested low-rank objective of Ryu et al. (ICML 2024) [1]. The direct application of this framework to stochastic Koopman operators, including post-training calibration and prediction, is established prior work by Jeong et al. (NeurIPS 2025) [2]. This repository is an implementation and experimental integration of those ideas, not a claim that this application is new.

The modern implementation is independent of the historical scripts retained elsewhere in this repository. It does not reproduce all experiments or theoretical results of either paper. The controller paper [4] is relevant background for future controlled-system work; the present identification model does **not** implement its robust controller or inherit its stability guarantees.

## 1. What operator is identified?

Let `(X, Y)` be a random lagged state pair with joint law `rho01`, source marginal `rho0`, and target marginal `rho1`. For a time-homogeneous Markov process, `Y` is the state one chosen lag after `X`. Define

\[
 (\mathcal K h)(x)=\mathbb E[h(Y)\mid X=x],\qquad
 \mathcal K:L^2(\rho_1)\longrightarrow L^2(\rho_0).
\]

The operator acts **backward on observables** even though it predicts future measurements. A left singular function is a function of the source state; a right singular function is a function of the target state:

\[
 \mathcal K\psi_i=\sigma_i\phi_i,\qquad
 \mathcal K^*\phi_i=\sigma_i\psi_i,
 \quad
 \langle\phi_i,\phi_j\rangle_{\rho_0}
 =\langle\psi_i,\psi_j\rangle_{\rho_1}=\delta_{ij}.
\]

Here `left_functions(x)` evaluates `phi`, and `right_functions(y)` evaluates `psi`. Swapping these conventions can silently identify the adjoint instead of the desired forward prediction operator.

Jensen's inequality gives

\[
 \|\mathcal Kh\|_{\rho_0}^2
 \leq \mathbb E\{|h(Y)|^2\}=\|h\|_{\rho_1}^2.
\]

Thus a conditional-expectation Koopman operator is a contraction. Its singular values, when defined, are at most one. Since both marginals are probability measures, `phi_0=psi_0=1` is a unit singular pair. The model reserves one of its `rank` outputs for that constant by default. This does not assume stationarity or reversibility. Other singular values can also equal one, so the leading singular function need not be uniquely identifiable.

For a transition density `p(y|x)`, the integral kernel relative to the **target marginal** is

\[
 k(x,y)=\frac{p(y\mid x)}{\rho_1(y)},\qquad
 (\mathcal Kh)(x)=\int k(x,y)h(y)\rho_1(dy),
\]

where a density ratio exists. Replacing `rho1` with Lebesgue measure, a uniform distribution, or `rho0` changes the operator geometry unless justified separately.

## 2. Neural training objective

Let `f_theta(x), g_eta(y)` be column vectors with `r` outputs. These are **scaled factors**, not initially normalized singular functions. Write

\[
 C_f=\mathbb E[f(X)f(X)^T],\quad
 C_g=\mathbb E[g(Y)g(Y)^T],\quad
 C_{fg}=\mathbb E[f(X)g(Y)^T].
\]

The low-rank kernel is `k_r(x,y)=f(x)^T g(y)`. Its loss is

\[
 \mathcal L_r=-2\operatorname{tr}(C_{fg})
              +\operatorname{tr}(C_fC_g).
\]

To see the origin of both terms, expand the squared Hilbert--Schmidt error:

\[
 \|\mathcal K-\widehat{\mathcal K}\|_{HS}^2
 =\|\mathcal K\|_{HS}^2
 -2\sum_i\mathbb E[f_i(X)g_i(Y)]
 +\sum_{i,j}\mathbb E[f_i(X)f_j(X)]
                    \mathbb E[g_i(Y)g_j(Y)].
\]

The first term is independent of the networks. This finite-error identity requires `K` to be Hilbert--Schmidt: compactness alone does not ensure a finite Hilbert--Schmidt norm. The polynomial loss itself remains meaningful for bounded operators and square-integrable finite feature families even when that constant is infinite.

The unconstrained rank-`r` loss identifies a subspace; its factorization has a change-of-basis ambiguity. To prefer ordered prefixes, use positive prefix weights `w_l`:

\[
 \mathcal L_{nested}=\sum_{\ell=1}^{r}w_\ell\mathcal L_\ell.
\]

Set `a_i=sum_{l=i}^r w_l` and `W_ij=a_max(i,j)` with one-based indices. The same objective can be evaluated without a loop over prefixes:

\[
 \mathcal L_{nested}
 =-2\sum_i a_i(C_{fg})_{ii}
   +\sum_{i,j}W_{ij}(C_f)_{ij}(C_g)_{ij}.
\]

Uniform `w_l=1/r` gives `a_i=(r-i+1)/r`. A zero last prefix weight would remove the guarantee for the last mode, so indexing matters. Joint nesting permits a multi-output network with shared hidden layers [1]. Distinct singular values permit identification of individual modes up to sign and reciprocal factor scaling; repeated singular values only identify their singular subspaces.

No SVD, matrix inverse, or eigendecomposition is needed in this training objective. All neural parameters receive ordinary automatic-differentiation gradients. Consequently, this is not a fixed-feature SVD relabeled as neural learning.

### Sampling and bias

The alignment term needs **paired** samples. The two marginal Gram matrices in the product need independent estimates to obtain an unbiased stochastic estimate of their product. With one paired minibatch of size `B`,

\[
 \mathbb E[\widehat C_{f,ij}\widehat C_{g,ij}]
 = C_{f,ij}C_{g,ij}
   +\frac{\operatorname{Cov}(f_i(X)f_j(X),g_i(Y)g_j(Y))}{B}
\]

for IID pairs when the displayed covariance is finite. Using the same paired minibatch without correction therefore adds bias. Independent source and target index draws, **with replacement**, instead give an unbiased estimate of the product of empirical marginal moments conditional on the fixed training dataset. That statement is about optimization of an empirical objective; it does not imply that a finite, correlated trajectory provides an unbiased estimate of every population quantity. Independent trajectories, separated time blocks, or an explicit dependence analysis are needed for population uncertainty statements.

## 3. Calibration after neural learning

Networks are calibrated using training data only. With the constant retained, centering nonconstant outputs under their respective marginals preserves the represented function spaces. In the ideal nonsingular case, form

\[
 M=C_f^{-1/2}C_{fg}C_g^{-1/2}=U\Sigma V^T,
\]

and define

\[
 \phi(x)=U^TC_f^{-1/2}f(x),\qquad
 \psi(y)=V^TC_g^{-1/2}g(y).
\]

This small SVD is a **post-training calibration inside the learned spaces**. It does not replace neural optimization. It gives the best empirical two-space approximation supported by those spaces. This implementation applies a relative Gram eigenvalue cutoff and discards poorly determined directions. The whitening matrices can therefore be rectangular. Calibration reports the requested rank, source and target effective ranks, and retained rank; calibrated functions can have fewer columns than the raw network outputs. Orthonormality holds inside the retained empirical spaces up to numerical precision. The fixed constant is preserved rather than regularized away.

Raw factor norm products are also exposed for inspection, but they are singular-value estimates only when the nested factors have converged to appropriately orthogonal singular directions. Their differences from the calibrated spectrum, and the raw orthogonality diagnostics, help reveal when post-training calibration is compensating for incomplete neural convergence.

For row-oriented feature arrays, the same transformations are `F @ C_f^(-1/2) @ U` and `G @ C_g^(-1/2) @ V`. Source normalization and source centering must also be used when evaluating `phi` at a target-state input.

A validation or test batch must not refit these transformations. A held-out empirical singular value or cross-correlation can exceed one through sampling error or distribution shift; clipping it silently would hide that information.

## 4. Prediction and iteration

For an observable vector `h`, define

\[
 b_h=\mathbb E_{\rho_1}[\psi(Y)h(Y)^T].
\]

The calibrated one-step predictor is

\[
 (\widehat{\mathcal K}h)(x)=\phi(x)^T\Sigma b_h.
\]

Choosing `h(y)=y` predicts the conditional mean of the next state, in the original state coordinates. Predicting a random realization perfectly is not possible in general. Indeed,

\[
 \mathbb E\|Y-\widehat m(X)\|^2
 =\mathbb E\|\mathbb E[Y\mid X]-\widehat m(X)\|^2
  +\mathbb E\operatorname{tr}\operatorname{Cov}(Y\mid X).
\]

Synthetic experiments should distinguish error against the known conditional mean from error against noisy next-state observations.

The left and right singular bases generally differ. To iterate the approximate operator, evaluate both bases at the **same target-state inputs** and form

\[
 H=\mathbb E_{\rho_1}[\psi(Y)\phi(Y)^T].
\]

Direct substitution into the integral operator gives, for `n >= 1`,

\[
 (\widehat{\mathcal K}^{\,n}h)(x)
 =\phi(x)^T\Sigma(H\Sigma)^{n-1}b_h.
\]

This is why neither `Sigma**n` nor a lagged correlation `E[psi(Y)phi(X)^T]` is a correct replacement for the bridge matrix. Nonzero eigenvalues of the finite-rank operator are eigenvalues of `Sigma @ H` (equivalently `H @ Sigma`), **not** its singular values. Negative and complex eigenvalues can occur even though singular values are nonnegative.

The formula iterates the learned approximation. It does not make `K_hat**n` equal to the best rank-`r` approximation of `K**n`. For population-level composition, source and target must share a state domain and the source functions must be evaluable and square-integrable under the target marginal. With nonstationary sampling, extrapolation to later distributions is an additional assumption. Long-horizon accuracy can fail even when one-step error is small. A neural-observable EDMD predictor is a separate useful method; it must be labeled separately from the singular-factor predictor above.

## 5. Deterministic systems require special care

If `Y=F(X)` and `rho1` is the pushforward of `rho0`, then

\[
 \|\mathcal Kh\|_{\rho_0}^2
 =\int |h(F(x))|^2\rho_0(dx)
 =\int |h(y)|^2\rho_1(dy).
\]

Thus `K` is an isometry, and `K* K=I`. On an infinite-dimensional domain, an isometry is not compact: it maps an orthonormal sequence to another orthonormal sequence, which has no norm-convergent subsequence. Its modulus has singular value one in every direction. For any finite-rank `R`, there is a unit vector in its kernel and therefore `||K-R|| >= 1`. If the dynamics are invertible and preserve the chosen measure, `K` is unitary.

Consequently, the unregularized deterministic logistic map does not have a decaying compact Koopman SVD in this natural `L2` geometry. A finite-dictionary matrix can still have a perfectly useful SVD, but it represents a projected operator, not convergence to a preferred sequence of full-operator singular modes. Adding process noise or smoothing may produce a compact or Hilbert--Schmidt operator; it also changes the identification target. Nonzero noise alone is not a proof of compactness on every state space. Finite-state or other finite-dimensional degenerate deterministic cases are exceptions to the infinite-dimensional argument.

The original compact-operator assumptions, the role of Markov dynamics, and the use of variational singular subspaces should therefore remain explicit [1--3]. Neural optimization also has no general guarantee of finding the global minimum in a restricted nonconvex model class.

## 6. Exact reference problems and interpretation of accuracy

Useful checks separate errors in implementation, sampling, optimization, and approximation:

* **Finite Markov chain:** for row-stochastic `P`, source probabilities `rho0`, and `rho1=rho0 @ P`, the correct Euclidean reference matrix is `diag(sqrt(rho0)) @ P @ diag(1/sqrt(rho1))` on positive-mass states. An ordinary unweighted SVD of `P` is generally a different problem. Full-rank reconstruction must recover `P`, and composition must recover `P**n`.
* **Stationary Gaussian AR(1):** `Y=a X+sqrt(1-a*a) epsilon`, with independent standard-normal `X,epsilon` and `0<a<1`, has normalized probabilists' Hermite functions and singular values `1,a,a*a,...`. The conditional-mean prediction is `a**n * x`. This gives a stochastic, nonlinear-observable spectral check rather than only a state-regression check.
* **Nonnormal Gaussian linear dynamics:** choose `Y=A X+epsilon` with `X~N(0,I)` and noise covariance `I-A A^T` positive definite. If `A=U S V^T`, the linear right/input singular functions use `U`, while left/output singular functions use `V`: `K(u_i^T y)=s_i v_i^T x`. This catches the transpose error hidden by symmetric examples. The complete spectrum also contains products associated with higher-order Gaussian observables; the matrix singular values alone are not the entire Koopman spectrum.

Keep train, validation, and test observations separate. Use validation data for checkpoint selection and hyperparameters, then report test errors once. For trajectories, split by independent trajectories or separated time blocks rather than randomly interleaving adjacent pairs. Test approximate orthonormality, cross-mode alignment, constant preservation, operator orientation, multi-step composition, and serialization. A single small seeded experiment establishes feasibility for that setting; it does not establish superiority over EDMD, universal accuracy, or performance on high-dimensional physical systems.

## References

1. J. Jon Ryu, Xiangxiang Xu, H. S. Melihcan Erol, Yuheng Bu, Lizhong Zheng, Gregory W. Wornell. **Operator SVD with Neural Networks via Nested Low-Rank Approximation.** ICML 2024, PMLR 235:42870--42905. [Paper](https://proceedings.mlr.press/v235/ryu24b.html), [arXiv](https://arxiv.org/abs/2402.03655), [original code](https://github.com/jongharyu/neural-svd). In particular, Sections 3.1--3.3 describe the low-rank objective and nesting.
2. Minchan Jeong, J. Jon Ryu, Se-Young Yun, Gregory W. Wornell. **Efficient Parametric SVD of Koopman Operator for Stochastic Dynamical Systems.** NeurIPS 2025. [Proceedings](https://papers.nips.cc/paper_files/paper/2025/hash/24826a1c623fe41706850748f3dc7370-Abstract-Conference.html), [arXiv](https://arxiv.org/abs/2507.07222), [official code](https://github.com/MinchanJeong/NeuralKoopmanSVD). Sections 2--3 and Table 1 discuss the stochastic operator, training, and inference.
3. Hao Wu, Frank Noe. **Variational approach for learning Markov processes from time series data.** Journal of Nonlinear Science 30, 23--66 (2020). [Preprint](https://arxiv.org/abs/1707.04659), [DOI](https://doi.org/10.1007/s00332-019-09567-y). VAMP provides the variational singular-subspace context; its Hilbert--Schmidt assumptions are material.
4. Robin Strasser, Manuel Schaller, Karl Worthmann, Julian Berberich, Frank Allgower. **Koopman-based feedback design with stability guarantees.** IEEE Transactions on Automatic Control 70(1), 355--370 (2025). [Preprint](https://arxiv.org/abs/2312.01441), [DOI](https://doi.org/10.1109/TAC.2024.3425770).
