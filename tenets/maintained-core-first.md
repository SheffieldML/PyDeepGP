---
conflicts_with: []
created: '2026-11-13'
id: maintained-core-first
last_reviewed: '2026-11-13'
last_updated: '2026-11-13'
review_frequency: Annual
status: Active
tags:
- tenet
- prioritization
- maintenance
- core
title: Maintained Core First
version: '1.0'
---

# Maintained Core First

## Tenet

**Title**: Maintained Core First

**Description**: The serial sparse-variational DeepGP path—supervised and unsupervised stacks using VarDTC / SVI-VarDTC without MPI or MRD—is the product we maintain. Optional or historical paths (MPI parallel inference, MRD views, Theano encoders, prediction-layer clones) may remain in the tree, but they do not set the quality bar until they are revived with tests and docs.

**Quote**: *"Keep the spine strong before polishing the ribs"*

**Examples**:
- Gating CI coverage on the maintained serial DeepGP surface
- Prioritizing bugfixes in `DeepGP.predict`, `optimize`, and layer `checkgrad` over MRD plot helpers
- Accepting PRs that harden core construction and inference before expanding exotic backends
- Triaging open issues: FAQ and core API first; MRD recursion later

**Counter-examples**:
- Blocking a GPy-compatibility release because unused MPI code fails lint
- Spending a release cycle on MRD plot_latent while core prediction is undertested
- Letting coverage targets force investment in dead modules instead of the spine
- Treating every historical code path as equally supported

**Conflicts**:
- *Conflict*: Users who still need MRD or MPI vs. focus on the serial core
- *Resolution*: Document the gap; accept revival CIPs when someone owns tests and maintenance
- *Conflict*: Deleting deferred code vs. keeping it for archaeology
- *Resolution*: Prefer keep-but-omit from the quality gate until a revival plan exists
