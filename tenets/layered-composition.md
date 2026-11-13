---
conflicts_with: []
created: '2026-11-13'
id: layered-composition
last_reviewed: '2026-11-13'
last_updated: '2026-11-13'
review_frequency: Annual
status: Active
tags:
- tenet
- architecture
- layers
- composition
title: Layered Composition
version: '1.0'
---

# Layered Composition

## Tenet

**Title**: Layered Composition

**Description**: A Deep GP is a composition of GP layers with clear roles: observed, hidden, and top. Dimension lists, per-layer kernels, and inducing-point counts are the primary configuration surface. New modelling power should appear as new or better-behaved layers that compose into `DeepGP`, not as a separate modelling dialect.

**Quote**: *"Depth is a stack of GPs, not a new language"*

**Examples**:
- Building models with `nDims` / kernel lists that map one-to-one onto layers
- Adding depth via `add_layer` on an existing DeepGP
- Keeping inference methods pluggable per layer (`VarDTC`, `SVI`) while sharing the layer interface
- Examples that show supervised and unsupervised stacks with the same constructor pattern

**Counter-examples**:
- Special-casing depth in a monolithic class that bypasses the layer API
- Requiring users to wire inference objects by hand for the default supervised case
- Introducing a parallel "deep model" type that does not reuse `ObservedLayer` / `HiddenLayer`
- Making intermediate latent shapes implicit and undocumented

**Conflicts**:
- *Conflict*: Convenience one-shot constructors vs. explicit layer control
- *Resolution*: Keep `DeepGP(...)` as the high-level builder; expose layers for advanced control
- *Conflict*: Global inference settings vs. per-layer specialization
- *Resolution*: Sensible defaults at construction; allow per-layer overrides where needed
