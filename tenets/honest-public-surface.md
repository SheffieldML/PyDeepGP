---
conflicts_with: []
created: '2026-11-13'
id: honest-public-surface
last_reviewed: '2026-11-13'
last_updated: '2026-11-13'
review_frequency: Annual
status: Active
tags:
- tenet
- api
- honesty
- documentation
title: Honest Public Surface
version: '1.0'
---

# Honest Public Surface

## Tenet

**Title**: Honest Public Surface

**Description**: The public API and docs must describe what this repository actually ships. Entry points that depend on missing modules, abandoned backends, or unmaintained optional paths should fail clearly or be omitted—not present as if they work. Prefer a smaller, truthful surface over a larger, broken one.

**Quote**: *"If it is not in the tree, do not pretend it is in the product"*

**Examples**:
- Raising `NotImplementedError` from `gen_pred_model` / `gen_pred_layer` when PredLayer modules are absent
- Skipping Theano back-constraint tests when Theano is not installed
- Documenting that MPI, MRD, and prediction-only SVI helpers are deferred in README and coverage config
- Closing issues that ask for missing modules by stating they are not shipped

**Counter-examples**:
- Leaving imports that explode with opaque `ImportError` from nonexistent packages
- README claims for Variational Auto-encoded Deep GPs when the encoder path is Theano-only and untested
- Keeping dead methods that look usable in autocomplete but cannot run
- Measuring CI success while silently skipping the only tests that exercise advertised features

**Conflicts**:
- *Conflict*: Historical API names users may still call vs. removing broken methods
- *Resolution*: Keep the name, fail with an explicit message and a pointer to supported alternatives
- *Conflict*: Research completeness vs. honesty about maintenance
- *Resolution*: Mark research paths as deferred in docs and coverage; do not block the maintained core
