---
id: "2026-11-27_pure-layer-functions"
title: "Extract pure layer functions and define the layer interface"
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

# Task: Extract pure layer functions and define the layer interface

## Description

Phase 2 of CIP-0003. Move the layer bound, psi-statistic propagation and gradients into array-in / array-out functions (public but provisional; name `deepgp/functional/` provisional), following GPy CIP-000F Stage 2. `Layer` classes become a thin shell holding parameters. Define the per-layer interface that both inference families will implement: forward marginals at supplied inputs, accept upstream gradients, expose parameters.

## Acceptance Criteria

- [x] Functions take and return arrays only: no `Param`, observers or communicators
- [x] Layer classes call the functions; public class names and constructors unchanged
- [x] Objective and gradient identical to current values on the reference models (tolerance recorded)
- [x] `checkgrad` passes for every layer type
- [x] Interface documented in the module docstrings

## Related

- CIP: 0003 (Phase 2)
- GPy CIP-000F

## Progress Updates

### 2026-11-27

Task created from CIP-0003 (Accepted). Blocked on MPI removal.

### 2026-11-27 (first slice)

Added `deepgp/functional/`: `collapsed_bound` (statistics in, bound, posterior arrays and gradients w.r.t. statistics and outputs out; certain or uncertain outputs) and `route_gradients` (gradients w.r.t. statistics to kernel and inducing-input gradients). `VarDTC._inference_uncertain_outputs` and `SparseGPLayer._inference_vardtc` now call them; dead `psicov` branches removed. Reference tests unchanged (63 pass before, 70 now), so objective and gradients are identical to within the stored-value tolerances. `checkgrad` passes for unsupervised, supervised and back-constrained stacks; finite-difference tests cover the bound.

Honest limits: `route_gradients` still takes a GPy kernel object (GPy kernels are not pure until GPy CIP-000F Stage 2) and the shell assigns the returned arrays. The q(X) gradient routing in `Layer.update_qX_gradients`, MRD layers and the encoder layers are not yet extracted; the SVI path keeps its in-class code. Left open, so the task stays In Progress.
