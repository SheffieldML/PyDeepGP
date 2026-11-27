---
id: "2026-11-27_ard-scales-and-init"
title: "ARD scale inspection, pruning, and initialisation options"
status: "Completed"
priority: "Medium"
created: "2026-11-27"
last_updated: "2026-11-27"
category: "features"
related_cips: ["0003", "0002"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- matlab-parity
---

# Task: ARD scale inspection, pruning, and initialisation options

## Description

Phase 3 of CIP-0003. Recover the workflow of Matlab `hsvargplvmShowScales`, `PruneModel` / `RestorePrunedModel` and `InitXOptions` as small tested features: per-layer ARD relevance report, pruning of near-zero dimensions with restore, and named initialisers for `X` (PCA, per-view concatenation, random). We reproduce the capability, not the Matlab interface.

## Acceptance Criteria

- [x] Per-layer ARD scale report and plot helper
- [x] Prune and restore round-trip leaves the objective unchanged to tolerance
- [x] Initialiser selectable by name; default unchanged
- [x] Tests run in CI; the toy example from Damianou 2013 shows the relevant scales

## Related

- CIP: 0003 (Phase 3), 0002

## Progress Updates

### 2026-11-27

Task created from CIP-0003 (Accepted).

### 2026-11-27

Implemented in `deepgp/diagnostics.py` (`ard_scales`, `show_scales`, `prune_model`, `restore_model`) and `validate_init` / `INITIALISERS` in `deepgp/util/util.py`, with tests in `deepgp/testing/test_diagnostics.py` (85 passed overall).

Scope decisions:
- Pruning masks dimensions (kernel parameter set to its off value and fixed) rather than removing columns, so there is no speed-up. Restore is exact. Structural removal is not implemented.
- Unknown initialiser names used to fall through silently to random; they now raise `ValueError`. `random` is a new synonym for `rand`. The default (`PCA`) is unchanged.
- The toy example is a one-layer model on a 1-D curve, not the Damianou 2013 data; it shows one of three offered dimensions switched off (scales 1.00, 0.48, 0.01). Reproducing the paper's own figure belongs to Phase 5.
