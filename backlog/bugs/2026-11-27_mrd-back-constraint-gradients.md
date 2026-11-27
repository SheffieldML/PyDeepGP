---
id: "2026-11-27_mrd-back-constraint-gradients"
title: "Back-constrained MRD layer fails checkgrad (encoder and X_var gradients)"
status: "Completed"
priority: "High"
created: "2026-11-27"
last_updated: "2026-11-27"
category: "bugs"
related_cips: ["0003", "0002"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- bug
- mrd
---

# Bug: Back-constrained MRD layer fails checkgrad

## Description

Found while adding MRD `checkgrad` tests for CIP-0003 Phase 2. For `DeepGP([[3, 3], 2], [Y1, Y2], kernels=[[k1, k2]], num_inducing=4, back_constraint=True)`, `checkgrad` fails. It fails identically at commit 65dba79, before the q(X) extraction, so the extraction did not cause it. Non-back-constrained MRD (single layer and with a hidden layer) passes.

Observed with `m.checkgrad(verbose=True)`:

- Kernel, inducing-input and noise gradients are correct.
- Encoder (`view_i.mlp.*`) gradients: numerical is exactly 0.2 times analytical, for every weight (so the analytical gradient is 5x too large; 5 does not equal the number of views).
- `view_i.X_var` (shared variance): analytical is about 1e-6, numerical is large (about -272); the gradient is lost.
- `latent_space.mean/variance` are derived from the encoders in this mode and are overwritten on every update; their analytical gradients are non-zero but should be excluded (fixed) or ignored.

## Acceptance Criteria

- [ ] `MRDCheckgradTest.test_mrd_back_constrained_checkgrad` passes without `expectedFailure`
- [ ] Root cause recorded (candidates: `MRDView.scale` handling in `update_gradients`, `update_encoder_gradients` ordering against `X_var_common`, and `latent_space` being a free parameter)

## Related

- CIP: 0003 (Phase 3: MRD), 0002
- Test: `deepgp/testing/test_functional.py`

## Progress Updates

### 2026-11-27

Found; test marked `expectedFailure` so CI records it without blocking the maintained path.

## Progress Updates

### 2026-11-27
Fixed. Two independent causes:

1. `MLP.update_gradient` accumulated into the layer gradient buffers, which were only reset by `predict`. Each repeated `parameters_changed` between two `predict` calls added another copy of the encoder gradient, so the analytic gradient was an integer multiple of the true one (5x in the repro). `update_gradient` now resets the buffers first and is idempotent.
2. In back-constrained mode the aggregate q(X) (`latent_space`) is derived from the view encoders on every update but was a free parameter, so `checkgrad` (and any optimiser) saw nonzero analytic gradients against zero numerical ones. It is now fixed.

The `X_var` gradients were correct throughout. `test_mrd_back_constrained_checkgrad` is no longer an expected failure; full suite 75 passed.
