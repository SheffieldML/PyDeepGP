---
id: "2026-11-27_share-vardtc-with-gpy"
title: "Replace local VarDTC with GPy's, including CIP-0013 sparse linalg modes"
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

# Task: Replace local VarDTC with GPy's

## Description

Phase 2 of CIP-0003. After MPI is removed, `deepgp/inference/vardtc.py` differs from GPy's `VarDTC` mainly by the uncertain-input handling and chunking. Work out the real delta, upstream what belongs in GPy (or subclass thinly), and delete the rest. Pick up GPy CIP-0013's `stable` / `lq` linear algebra modes where the layer bound uses them.

## Acceptance Criteria

- [ ] A written comparison of `deepgp.inference.vardtc.VarDTC` against `GPy...VarDTC` (what differs, and why)
- [ ] Local code reduced to the documented delta
- [ ] Reference-value tests unchanged
- [ ] Choice of linalg mode exposed per layer, defaulting to current behaviour

## Related

- CIP: 0003 (Phase 2)
- GPy CIP-0013

## Progress Updates

### 2026-11-27

Task created from CIP-0003 (Accepted). Blocked on MPI removal.
