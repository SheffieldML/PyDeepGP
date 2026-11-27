---
id: "2026-11-27_ard-scales-and-init"
title: "ARD scale inspection, pruning, and initialisation options"
status: "Proposed"
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

- [ ] Per-layer ARD scale report and plot helper
- [ ] Prune and restore round-trip leaves the objective unchanged to tolerance
- [ ] Initialiser selectable by name; default unchanged
- [ ] Tests run in CI; the toy example from Damianou 2013 shows the relevant scales

## Related

- CIP: 0003 (Phase 3), 0002

## Progress Updates

### 2026-11-27

Task created from CIP-0003 (Accepted).
