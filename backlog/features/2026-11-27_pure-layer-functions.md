---
id: "2026-11-27_pure-layer-functions"
title: "Extract pure layer functions and define the layer interface"
status: "Proposed"
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

- [ ] Functions take and return arrays only: no `Param`, observers or communicators
- [ ] Layer classes call the functions; public class names and constructors unchanged
- [ ] Objective and gradient identical to current values on the reference models (tolerance recorded)
- [ ] `checkgrad` passes for every layer type
- [ ] Interface documented in the module docstrings

## Related

- CIP: 0003 (Phase 2)
- GPy CIP-000F

## Progress Updates

### 2026-11-27

Task created from CIP-0003 (Accepted). Blocked on MPI removal.
