---
id: "2026-11-27_share-vardtc-with-gpy"
title: "Replace local VarDTC with GPy's, including CIP-0013 sparse linalg modes"
status: "In Progress"
priority: "Medium"
created: "2026-11-27"
last_updated: "2026-11-27"
category: "features"
related_cips: ["0003"]
owner: "Neil Lawrence"
dependencies: ["2026-11-27_remove-mpi"]
tags:
- backlog
- refactor
---

# Task: Replace local VarDTC with GPy's

## Description

Phase 2 of CIP-0003. After MPI is removed, `deepgp/inference/vardtc.py` differs from GPy's `VarDTC` mainly by the uncertain-input handling and chunking. Work out the real delta, upstream what belongs in GPy (or subclass thinly), and delete the rest. Pick up GPy CIP-0013's `stable` / `lq` linear algebra modes where the layer bound uses them.

## Acceptance Criteria

- [x] A written comparison of `deepgp.inference.vardtc.VarDTC` against `GPy...VarDTC` (what differs, and why)
- [x] Local code reduced to the documented delta
- [x] Reference-value tests unchanged
- [x] Choice of linalg mode exposed per layer, defaulting to current behaviour (`sparse_linalg=` on `SparseGPLayer`; settable later via `layer.inference_method.sparse_linalg`)

## Related

- CIP: 0003 (Phase 2)
- GPy CIP-0013

## Progress Updates

### 2026-11-27

Task created from CIP-0003 (Accepted). Blocked on MPI removal.

### 2026-11-27 (comparison and result)

Compared on identical inputs (certain and uncertain X, Gaussian noise 0.1): log marginal, all gradients and the woodbury posterior agree with GPy's `VarDTC` to about 1e-11 once `const_jitter` is matched. The algebra was never the difference. The real delta:

| Feature | GPy `VarDTC` | PyDeepGP (before) | Now |
|---|---|---|---|
| Uncertain outputs (`Y` is q(X) of layer below) | not supported | supported | local code, `_inference_uncertain_outputs` (the only retained algebra) |
| Chunked psi-statistics | no (psi arguments can be passed in) | yes | `psi_statistics`, passed to GPy via `psi0/psi1/psi2` |
| `Kuu_sigma` (hidden layers) | no (`Lm` can be passed in) | yes | factor built locally, passed as `Lm` |
| Jitter / noise floor | 1e-8 | 1e-6 | `const_jitter = 1e-6`, floor 1e-6 (reference tests depend on it) |
| Heteroscedastic noise, mean function, caching, `to_dict`, `sparse_linalg` | yes | no | inherited |
| `inferentia` option | n/a | imported a non-existent class | raises `NotImplementedError` |

`vardtc.py` still carries the uncertain-outputs derivation. Moving it into GPy as an `uncertain_outputs` capability is the upstreaming candidate; it is left for a GPy CIP rather than done here. The linalg mode is exposed on `SparseGPLayer`; `DeepGP`-level pass-through is not yet wired. Tests: `deepgp/testing/test_vardtc_gpy.py` (63 pass overall, reference tests unchanged).
