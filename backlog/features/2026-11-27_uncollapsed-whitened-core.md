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
- [ ] `DeepGP`-level shell with inference selectable by name; default unchanged
- [ ] Minibatching with a seeded generator
- [ ] Reproduce GPc CIP-0035 reference numbers on a shared small problem (Phase 5 overlap)

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
