# Certifying a forecast and comparing it to a baseline

This module answers two questions for a **bounded scalar observable**: how large can a fitted predictor's conditional-mean error be, and can we establish that one predictor improves on a fixed baseline? The answers concern initial-state-distribution averaged risk at specified times. They do not bound a realized trajectory's deviation pointwise.

## Sampling contract

Condition on the complete fitting and validation data. Freeze the candidates (including dictionaries, estimated bridges, hyperparameters and ranks), the forecast horizons, the scalar observable, its *known* bounds $[a,b]$, the calibration sample size and the confidence level. Clip every predictor to $[a,b]$, including at deployment. The unbounded predictor has no certificate from this implementation. Do not estimate the range from calibration extrema. For an unbounded state, use a bounded physical observable or state explicitly that the target is transformed.

Draw $n\ge2$ **independent starts** from the intended evaluation law. At each start $X_i$, generate a trajectory and, for a direct conditional-mean risk estimate, a second trajectory that is independent and identically distributed conditional on $X_i$. At horizon $h$, call the two endpoints $Y_{ih},Y'_{ih}$. Their transitions may change with time; the branches must obey the same prescribed transition laws. Shared latent random forcing breaks the replication assumption unless it is part of the conditioning information. A single branch can supply multiple horizons. The statistical unit is the start, **not** all transitions cut from a few trajectories. If horizons use different initial laws, each column still needs its own i.i.d. units from its declared law. Train and calibrate on disjoint trajectories. A new deployment law needs a new argument or calibration sample.

## Absolute risk from paired continuations

For a frozen predictor $p_{jh}$, let $m_h(x)=E[Y_h\mid X=x]$ and $R_{jh}=E[(p_{jh}(X)-m_h(X))^2]$. Define

$$
Z_{ijh}=(p_{jh}(X_i)-Y_{ih})(p_{jh}(X_i)-Y'_{ih}).
$$

Conditional independence gives $E[Z_{ijh}\mid X_i]=(p_{jh}(X_i)-m_h(X_i))^2$. Thus the signed sample mean estimates the **conditional-mean MSE**, without adding process noise. Do not replace each signed product or its sample mean by its positive part before computing the concentration radius. This is an application of the replicated-successor principle; [Colbrook et al., *Beyond expectations*](https://doi.org/10.1007/s11071-023-09135-w) use batched snapshots to separate stochastic Koopman residuals and variance.

With $D=b-a$, the score range is $[-D^2/4,D^2]$: the most negative product at a fixed prediction $p$ is $-(p-a)(b-p)$, and $(p-a)(b-p)\le D^2/4$. Its range diameter is $d=5D^2/4$. Let $M=JH$ for $J$ candidates and $H$ horizons. Fix the concentration method before calibration. The one-sided radius is either

$$
s=d\sqrt{\log(M/\delta)/(2n)} \qquad\text{(Hoeffding)}
$$

or, with unbiased sample variance $v$ and $t=\log(2M/\delta)$,

$$
s=\sqrt{2vt/n}+7dt/[3(n-1)] \qquad\text{(empirical Bernstein)}.
$$

Rescale the score to unit range, apply the chosen inequality to each cell, and sum failure probabilities. See [Maurer and Pontil, Theorem 4 and Corollary 5](https://arxiv.org/pdf/0907.3740). With probability at least $1-\delta$, every risk is at most $U_{jh}=\operatorname{clip}(\bar Z_{jh}+s_{jh},0,D^2)$. Clamping the **final** upper bound uses the known deterministic range of the risk and does not modify the variance estimate. The implementation computes the radii in normalized units for numerical stability and returns bounds in squared observable units.

If only one continuation is available, substitute $Z=(p-Y)^2$, with $d=D^2$. Its mean is $R_{jh}+E[\mathrm{Var}(Y_h\mid X)]$, so the procedure still upper-bounds $R_{jh}$. The sample mean in this mode estimates ordinary noisy prediction MSE; it may remain high even for an exact conditional predictor.

## Comparing models without replicates

For a fixed baseline forecast $b_h(x)$, use the same observed outcome for both predictors:

$$
G_{ijh}=(b_h(X_i)-Y_{ih})^2-(p_{jh}(X_i)-Y_{ih})^2 .
$$

Its expectation equals $R_{bh}-R_{jh}$ because conditional variance cancels. The code returns signed means and **lower** confidence bounds on the gain. Apply the same concentration result to $-G$, with score diameter $2D^2$ and menu size $(J-1)H$; the baseline's self-comparison is identically zero. A positive lower bound supports an improvement for that observable and horizon. A nonpositive bound is inconclusive. The result assumes both models see the same initial information and use the same endpoint observable. It does not assert better singular-value recovery.

## Selection and interpretation

Maximize the length of the initial sequence of horizons satisfying $\sqrt{U_{jh}}\le\epsilon$, for an absolute RMSE tolerance $\epsilon$. Ties prefer smaller rank and then earlier menu position. Abstain if no first horizon is certified. A failure at $h=2$ prevents a certified prefix extending through $h=3$, even if the third isolated bound is small. Use horizons $1,\ldots,H$ for a physical forecast horizon; with irregularly spaced horizons the result is a count of evaluated positions only.

Validity survives rank/horizon selection because all cells are bounded simultaneously. A new method devised after inspecting calibration, or returning the better of two concentration methods chosen on calibration, is outside that guarantee. If several datasets, absolute bounds and relative improvements are advertised as jointly covered, allocate failure probabilities to sum to the desired total; otherwise label them as separate confidence statements. Repeatedly increasing $n$ until a bound passes is outside this fixed-sample result.

These are end-to-end, observable-specific certificates: errors from fitting dictionaries, calibrating singular functions and estimating cross-time bridges enter the final predictor's risk; their individual sizes are not identified or separately bounded. We do not certify the full Koopman operator norm, individual singular functions, closed-loop stability, a specific initial state's risk, arbitrary distribution shifts, or forecasts along one dependent trajectory. A wide bound can force abstention. Use independent test trajectories or analytically known conditional means to evaluate actual error, sharpness and coverage without retuning the menu.

## API

```python
from koopman_svd.certification import certify_forecasts, certify_improvement, select_horizon

# forecasts.shape == (candidate, horizon, independent_start)
# outcomes.shape == replicas.shape == (horizon, independent_start)
absolute = certify_forecasts(forecasts, outcomes, replicas, bounds=(-1, 1), delta=0.025)
chosen = select_horizon(absolute, ranks=ranks, tolerance=0.30)
contrast = certify_improvement(forecasts, outcomes, baseline=0, bounds=(-1, 1), delta=0.025)
# Two separate 97.5% statements imply at least 95% joint confidence for this evaluation.
```

## Optional tighter gain bounds when the entire start support is known

The generic paired-score interval has diameter $2(b-a)^2$. A **fixed, exhaustive** table of model predictions on the entire support of the deployment start distribution can narrow it. Let $q(s)$ denote the clipped baseline and $p(s)$ the clipped candidate at one horizon. For each possible start $s$, the gain score
$$
g(s,y)=(q(s)-y)^2-(p(s)-y)^2=(q(s)-p(s))(q(s)+p(s)-2y)
$$
is affine in $y$. Compute the minimum $L$ and maximum $U$ over **all support states** and both $y=a,b$. Every possible gain score lies in $[L,U]$, without knowing the transition probabilities or conditional means. Replacing the global diameter by $U-L$ in the same concentration proof is therefore valid, conditional on the frozen fit and table. A conservative known superset of reachable starts is also valid. The variance term in empirical Bernstein does not change. This is an application of the standard bounded-score inequality, not a new concentration theorem.

Use `certify_improvement(..., support_predictions=table, support_indices=ids)`, where the table is shaped (candidate, horizon, support point). Indices can be one vector of i.i.d. starts shared across horizons or a (horizon, start) array. The code checks the observed predictions against the indexed entries after clipping; when numerical inference varies across batch sizes, deploy predictions by looking them up directly in the frozen table. **Matching sampled rows does not establish completeness.** Never use the unique sampled starts, extrema of observed gains, a discretization of continuous support without a proven enclosure, or a table estimated from calibration outcomes to justify a smaller diameter. With no externally established support use the default global range. The guarantee concerns the same observable, predictors and start law as the original gain bound.

## A support-free alternative: cap the change in the deployed forecast

A continuous state space does not provide an exhaustive list of attainable predictions. Instead, choose a maximum change $c$ **before calibration**, and deploy the clipped proposal $p=b+\mathrm{clip}(q-b,-c,c)$ where the baseline $b$ and proposal $q$ are already in the observable interval of width $D$. For $0\le c\le D$, the magnitudes of their residuals against any bounded endpoint differ by at most $c$ and are at most $D$. Hence
$$
\left|(b-Y)^2-(p-Y)^2\right|\le D^2-(D-c)^2=2Dc-c^2.
$$
Both extremes are attainable, giving score diameter $2(2Dc-c^2)$ instead of the generic $2D^2$. The original paired gain identity and finite-menu argument now apply using this smaller diameter. At $c=D$ the bound recovers the generic one; at a small $c$ it constrains the model itself, so the largest attainable gain also becomes smaller. No observed sample maximum or known finite state support enters the proof. Call `certify_capped_updates` on frozen proposals and prespecified caps, and actually deploy `cap_forecast` with the same cap. Certifying the original unconstrained proposal with this diameter is invalid.

## Testing a fixed operational tolerance

A uniform numerical upper bound is not always necessary: the operational question may be whether RMSE is below one prespecified tolerance $e$. The function `certify_rmse_threshold` applies a bounded-mean betting test (see [Waudby-Smith and Ramdas](https://doi.org/10.1093/jrsssb/qkad009)). It uses the same clipped forecasts, independent starts and optional conditional replicates as the absolute-risk estimator. Normalize the paired residual product by $D^2$ to give $Z_i\le1$; without replicas use squared endpoint errors, which conservatively include process variance. Write $t=(e/D)^2\in(0,1)$. For each fixed $f$ in {0.05, 0.15, 0.35, 0.60, 0.90}, form the product of $1+f(t-Z_i)/(1-t)$. Under the null that the conditional-mean risk is at least $e^2$, each factor is positive and has mean at most one. Independence across starts makes the product a valid e-value. Equally mixing the five products retains this property. With $M$ candidate–horizon cells, accept a forecast only when its mixture reaches $M/\delta$. Markov's inequality and a union bound give simultaneous false-acceptance probability at most $\delta$. A prefix must pass at every preceding horizon. The finite menu, tolerance, and probability allocation are fixed in advance. This test is an additional inferential procedure, not an upper confidence interval or a retrospective replacement for an experiment's primary gate.
