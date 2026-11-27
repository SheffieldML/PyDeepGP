---
id: "2026-11-27_remove-mpi"
title: "Remove MPI and SparseGP_MPI (CIP-0001 Phase 4, CIP-0002 Phase D)"
status: "In Progress"
priority: "High"
created: "2026-11-27"
last_updated: "2026-11-27"
category: "features"
related_cips: ["0003", "0001", "0002"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- refactor
- mpi
---

# Task: Remove MPI and SparseGP_MPI

## Description

Phase 2, step 1 of CIP-0003. Execute CIP-0001 R17–R24 and CIP-0002 Phase D: delete `inference/vardtc_parallel.py` and `util/parallel.py`; rename `SparseGP_MPI` to `SparseGPLayer` (keep a deprecated alias); strip root-only branches and reductions from layers, `enclayer.py`, `mrd.py`, `model.py` and the inference modules. Keep accepting `mpi_comm=None` and `mpi_root=0`; raise `TypeError` if `mpi_comm` is not `None`. Single-machine scale comes from `psi_chunk_size` / `psi_threads`.

## Acceptance Criteria

- [x] No `mpi4py` import or `mpi_comm` branch remains in `deepgp/`
- [x] `DeepGP(..., mpi_comm=comm)` with non-`None` raises `TypeError` naming the chunked-psi alternative
- [x] `SparseGP_MPI` still importable, with a `FutureWarning`, pointing at `SparseGPLayer`
- [x] Reference-value tests (`dgp_*.h5`, `dgp_bc_*.h5`) unchanged and passing
- [x] MPI parity and policy tests replaced by a test of the new error
- [x] `.coveragerc` and README no longer mention MPI as supported

## Implementation Notes

Removal follows CIP-0001's deprecation text (the `FutureWarning` has been on `main` since the Phase 2 merge). No multi-machine users were identified in the Discussion; if one appears, CIP-0001 allows reducing MPI to a reducer hook instead.

## Related

- CIP: 0003 (Phase 2), 0001, 0002

## Progress Updates

### 2026-11-27

Task created and started on branch `cip0003-remove-mpi`.

Implementation complete on `cip0003-remove-mpi`: 56 tests pass (53 before, plus the new removal tests); reference tests unchanged. Awaiting review before status moves to Completed.
