---
id: "2026-11-27_uncollapsed-whitened-core"
title: "Uncollapsed family: whitened marginals core and layer stack"
status: "In Progress"
priority: "High"
created: "2026-11-27"
last_updated: "2026-11-27"
category: "features"
related_cips: ["0003"]
owner: "Neil Lawrence"
dependencies: ["2026-11-27_pure-layer-functions"]
tags:
- backlog
- uncollapsed
---

# Task: Uncollapsed family: whitened marginals core and layer stack

## Description

Phase 4 of CIP-0003. GPy's SVGP cannot serve as a hidden layer (CIP-0003, Resolved Question 1), so PyDeepGP gets a small core of its own, as pure functions in `deepgp/functional/uncollapsed.py`:

1. Whitened marginals `(mu, var)` at supplied inputs, with a backward pass that takes upstream gradients and returns gradients for `q(u)`, kernel, `Z` and the inputs.
2. Whitened KL and its gradients.
3. Reparameterised sampling with a caller-supplied `eps` (so gradients are testable and results reproducible).
4. A layer stack (doubly stochastic bound) built on 1 to 3, then a `DeepGP`-shaped shell, minibatching and natural gradients.

## Acceptance Criteria

- [x] Marginals agree with a dense unwhitened computation
- [x] Backward pass passes finite-difference checks for every output
- [x] Gradients flow through a two-layer stack with frozen `eps`
- [x] Doubly stochastic bound, with a likelihood at the top, passes finite-difference gradient checks (the `checkgrad`-style test on the `DeepGP` shell comes with the shell)
- [x] `DeepGP`-level shell with inference selectable by name; default unchanged
- [x] Minibatching with a seeded generator
- [x] Reproduce GPc CIP-0035 reference numbers on a shared small problem (Phase 5 overlap)

## Related

- CIP: 0003 (Phase 4)
- Salimbeni & Deisenroth (2017)

## Progress Updates

### 2026-11-27

Task created and started. Slice 1: items 1 to 3 of the description.

Slice 1 done (`deepgp/functional/uncollapsed.py`, `deepgp/testing/test_uncollapsed.py`, 91 passed overall):
- `whitened_marginals` / `whitened_marginals_backward`, `whitened_kl`, `reparam_sample` / `reparam_sample_backward`, all array-in/array-out apart from the GPy kernel object.
- Marginals agree with a dense unwhitened computation; the backward pass matches finite differences for `q(u)`, kernel, `Z` and inputs (including a Cholesky backward pass for `Kmm`); gradients flow through a two-layer stack with frozen `eps`.
- Testing note: GPy kernels cache on array identity and parameter writes need the `Parameterized` API, so finite-difference checks pass copies and set kernel values with `kern[:] = ...`.

Next slice: the doubly stochastic bound over a stack (likelihood at the top, KL per layer), with a `checkgrad`-style test.

### 2026-11-27 (slice 2)

`deepgp/functional/doubly_stochastic.py` (`doubly_stochastic_bound`), tests in `deepgp/testing/test_doubly_stochastic.py` (95 passed overall):
- Gaussian likelihood at the top, per-layer whitened KL, reparameterised samples through hidden layers with caller-supplied `eps`, `scale` for minibatches, optional residual (identity) mean on hidden layers.
- With one layer the bound equals GPy `SVGP`'s `log_likelihood` (to 1e-4 relative; the difference is jitter handling) once `q(u)` is mapped between whitened and unwhitened forms.
- Finite-difference checks pass for `q(u)`, kernels, `Z`, inputs and the noise variance on a three-layer stack, with and without residual means, two samples.

Remaining: `DeepGP`-shaped shell with inference selectable by name, seeded minibatching, natural gradients, GPc cross-check. Non-Gaussian likelihoods at the top are not covered yet.

### 2026-11-27 (slice 3)

`deepgp/models/uncollapsed.py` (`DoublyStochasticDeepGP`, `WhitenedLayer`), tests in `deepgp/testing/test_doubly_stochastic_model.py` (104 passed overall):
- GPy `Model` holding kernels, `Z`, whitened `q(u)` (packed Cholesky as in GPy's SVGP) and the noise as parameters; `checkgrad` passes for stacks with and without residual means, with samples and on a minibatch.
- Randomness is explicit: frozen `eps` make the objective deterministic, so `optimize` and `checkgrad` work; `resample(seed)` and `set_batch` redraw. `fit_minibatch` runs Adam in GPy's unconstrained space with a seeded generator (identical trajectories for equal seeds; tested). `predict` is a Monte Carlo mixture over hidden layers.
- `DeepGP(..., inference_method='doubly_stochastic')` returns this model via `__new__`; the default path is unchanged (tested). It is a different model class, not a `DeepGP` instance.
- At initialisation `q(u)` equals the prior, so the top layer ignores its input and the bound does not depend on `eps` until `q(u)` moves; a test documents this.

Limitations: supervised only (no `q(X)`), Gaussian likelihood only, no natural gradients, Adam is hand-rolled and untuned, no comparison on a real dataset yet.
Remaining: natural gradients, unsupervised (latent `X`), GPc CIP-0035 cross-check.

### 2026-11-27 (slice 4: GPc cross-check)

`deepgp/testing/gpc_reference_generate.py` builds GPc `DeepGP` models (1, 2, 3 layers; N=20, M=4, hidden width 2), overwrites the packed stochastic parameter vector with random values (layout read from `CSvgp::getStochasticParams`: log kernel parameters, `Z`, log beta, then per output the means and the lower Cholesky with log diagonal), reloads, and records GPc's ELBO and `layer_kl`. The golden file `gpc_reference.json` is committed; `test_gpc_reference.py` evaluates the same parameters in PyDeepGP, so CI does not need GPc.
- One layer: ELBO agrees to 1e-4, KL to 1e-6, predictive mean and variance to 1e-5 (GPc RBF is `variance * exp(-0.5 * inverseWidth * r^2)`, so `lengthscale = inverseWidth^-0.5`; GPc beta is a precision).
- Two and three layers (Monte Carlo, GPc's `elbo()` averaged over 4000 calls): agreement within four combined standard errors; per-layer KL agree.
Not done: the V8 composite-task reproduction (RMSE <= 0.26, NLL <= 0) needs training and the GPc task generator; it remains open for Phase 5.

### 2026-11-27 (slice 5: natural gradients)

`whitened_expectation_grads` and `whitened_natgrad_step` in `deepgp/functional/uncollapsed.py`; the bound now also returns `natgrad` (`g_m`, `g_S` for the top layer, data term only). `DoublyStochasticDeepGP.fit_minibatch(..., natgrad_gamma=g)` updates the top layer's `q(u)` by `theta <- (1-g) theta + g (theta_prior + grad_eta ELL)` and leaves everything else on Adam (the hybrid scheme of Salimbeni, Eleftheriadis and Hensman 2018, as in GPc's `natgrad_gamma`). Tests (`test_natgrad.py`): with `gamma = 1` on the full data and one layer the step reaches the analytic optimum `S = (I + B B^T / noise)^-1`, `m = S B y / noise` and the `q(u)` gradients vanish; small steps increase the bound; `g_S` is consistent with the Cholesky gradient; hybrid training is deterministic for equal seeds.
Hidden layers stay on Adam. Not compared with GPc's trained trajectories.

### 2026-11-27 (slice 6: unsupervised)

`doubly_stochastic_bound(..., x_var, eps_x)` treats `X` as the mean of `q(X) = N(X, diag(x_var))`: each sample draws `X + sqrt(x_var) eps_x`, and `scale * KL(q(X) || N(0, I))` is subtracted (scaled by N/B with the data term, so minibatches stay unbiased). `DoublyStochasticDeepGP(nDims, Y, None, ...)` holds `X_mean` and `X_var` (log-exp constrained) as parameters, initialised by standardised PCA of `Y`, with frozen `eps_x` alongside `eps`. Tests: finite-difference gradients for `X` and `x_var` through a residual two-layer stack, tiny-variance limit equals the deterministic bound minus the latent KL, `checkgrad` on the full model and on a minibatch, and a short training run that improves the bound and predicts.
Limitations: Gaussian likelihood only; no back-constraint (amortised `q(X)`) and no MRD views for this model; natural gradients still only for the top layer; no real dataset (oil, CMU walk) comparison against the collapsed model yet.

### 2026-11-27 (slice 7: V8 and oil / CMU comparisons)

Scripts (not run in CI): `examples/composite_v8.py`, `examples/compare_collapsed_vs_doubly_stochastic.py`. `DoublyStochasticDeepGP` now defaults `X=None`, so `DeepGP(nDims, Y, inference_method='doubly_stochastic')` selects the unsupervised model.

**V8** (GPc composite task, RBF, M=64, hidden width 5, residual means, 3000 Adam steps, batch 256, lr 0.01, one seed, predictive NLL from the moment-matched Gaussian of 50 samples):

| L | test RMSE | test NLL | ELBO/N |
|---|-----------|----------|--------|
| 1 | 0.772 | 1.151 | -1.228 |
| 2 | 0.204 | -0.193 | -0.156 |
| 3 | 0.207 | -0.188 | -0.229 |

The gate (L=3: RMSE <= 0.26, NLL <= 0, beats one layer) is met. GPc's RBF numbers (3 seeds, same settings) are L=3 0.193 / -0.344 and L=2 0.249 / -0.171, so ours is within the RMSE range but the NLL is less sharp at L=3, and our L=2 is better than L=3 on this seed. One seed only; the ELBO at L=3 had not converged (it is below L=2). Initialisation (hidden-layer kernel variance, noise) is untuned and differs from GPc's `chi = 1/L` scheme.

**Oil** (100 points, 12-D standardised, `[12, 5, 2]`, M=30; collapsed VarDTC 5000 iterations, doubly stochastic 3000 full-batch Adam steps): 1-NN leave-one-out class error in the 2-D latent / reconstruction RMSE:

| model | seed 0 | seed 1 |
|-------|--------|--------|
| collapsed | 0.03 / 0.65 (500 iters) | 0.35 / 0.80 |
| doubly stochastic | 0.03 / 0.197 | 0.03 / 0.184 |
| + natgrad 0.1 | 0.03 / 0.178 | 0.03 / 0.172 |

**CMU** subject 35 walk 01 (358 frames, 62-D; train on even frames, predict odd frames from interpolated top latents, `[62, 5, 3]`, M=25; held-out RMSE on standardised data, predicting the mean gives 0.95): collapsed 0.438, doubly stochastic 0.281, with natgrad 0.265.

Caveats: single seeds; the collapsed model used the repository's example initialisation and is probably under-tuned (seed 1 landed in a poor optimum), so these runs do not show that doubly stochastic inference is better in general, only that it trains reliably here. Wall time is much longer for the doubly stochastic runs (100-870 s versus 8-225 s) because it is pure NumPy with per-sample Python loops. The CMU metric tests latent smoothness, not dynamics.

### 2026-11-27 (tuning slice 1: speed and seeds)

`fit_minibatch` recomputed the bound after every parameter write; it now defers that to the next
`resample`. A step is about 1.8x faster (profile: 0.97 s to 0.53 s for 20 steps, N=1000, M=64, two
layers), and `fit_minibatch` results are unchanged in kind (123 tests pass). The remaining cost is
mostly GPy kernel gradient routing, so a further gain needs batched kernel calls.

V8, 3000 iterations, batch 256, seeds 0 to 2 (test RMSE / NLL):

| L | seed 0 | seed 1 | seed 2 | seconds per run |
|---|--------|--------|--------|-----------------|
| 1 | 0.772 / 1.151 | 0.766 / 1.144 | 0.766 / 1.144 | 40 to 48 |
| 2 | 0.204 / -0.193 | 0.212 / -0.180 | 0.207 / -0.192 | 50 to 102 |
| 3 | 0.207 / -0.188 | 0.210 / -0.190 | 0.218 / -0.159 | 120 to 172 |

The gate is met on every seed, and the spread across seeds is about 0.01 in RMSE. A third layer
does not improve on two on this task. GPc-style initialisation (chi = 1/L) is not yet tried.

### 2026-11-27 (tuning slice 2: chi initialisation and a converged collapsed baseline)

`DoublyStochasticDeepGP(..., chi=c)` (also through `DeepGP(..., inference_method='doubly_stochastic')`)
starts hidden-layer kernels at variance `c/(L-1)`, GPc's CIP-0035 default scale. One test covers it.
The collapsed baseline in `examples/compare_collapsed_vs_doubly_stochastic.py` now restarts L-BFGS in
blocks of 1000 iterations, up to 5000, until the bound stops improving by 0.1%.

V8, `chi=1`, 3000 iterations, seeds 0 to 2 (test RMSE / NLL):

| L | seed 0 | seed 1 | seed 2 |
|---|--------|--------|--------|
| 2 | 0.204 / -0.193 | 0.212 / -0.180 | 0.207 / -0.192 |
| 3 | 0.235 / -0.091 | 0.215 / -0.151 | 0.199 / -0.185 |

At L=2, `chi/(L-1) = 1` is already our default, so the L=2 rows reproduce the earlier ones exactly.
At L=3 the hidden variance halves. It is worse on seed 0, better on seed 2, and the seed spread
(0.199 to 0.235) is larger than before (0.207 to 0.218). We see no benefit from `chi=1` on this task,
so the default stays `chi=None` (kernel variance 1).

Oil (`[12, 5, 2]`, M=30), seeds 0 to 2: leave-one-out 1-NN error / reconstruction RMSE.

| model | seed 0 | seed 1 | seed 2 |
|-------|--------|--------|--------|
| collapsed | 0.28 / 1.002 | 0.32 / 0.788 | 0.27 / 0.649 |
| doubly stochastic | 0.03 / 0.197 | 0.03 / 0.184 | 0.03 / 0.192 |
| + natgrad (0.1) | 0.03 / 0.178 | 0.03 / 0.172 | 0.03 / 0.182 |

CMU subject 35 walk, held-out RMSE, seed 0: collapsed 0.429, doubly stochastic 0.281, with natgrad
0.265; predicting the mean of the training frames gives 0.950.

Caveats. All hidden-layer `chi` settings here are no-ops for the two-layer oil and CMU stacks
(`chi/(L-1)=1`), so the `+ chi=1` variant equals `+ natgrad` and is not shown. The collapsed oil
model reaches a reconstruction RMSE near 1.0 on seed 0 (the level of predicting the mean), which
points to a poor local optimum; five thousand iterations from one initialisation per seed does not
rule out a better collapsed fit from other initialisations or noise settings. The oil and CMU
comparisons are three seeds and one seed respectively.
