---
conflicts_with: []
created: '2026-11-13'
id: gpy-native-stack
last_reviewed: '2026-11-13'
last_updated: '2026-11-13'
review_frequency: Annual
status: Active
tags:
- tenet
- gpy
- compatibility
- stack
title: GPy-Native Stack
version: '1.0'
---

# GPy-Native Stack

## Tenet

**Title**: GPy-Native Stack

**Description**: PyDeepGP is an extension of the GPy stack, not a parallel GP framework. Models, kernels, likelihoods, variational posteriors, and optimization should reuse GPy primitives wherever possible. Version and Python constraints track current GPy releases so Deep GP users inherit the same scientific Python baseline.

**Quote**: *"Deep GPs here are GPy models all the way down"*

**Examples**:
- Constructing layers with `GPy.kern`, `GPy.likelihoods`, and GPy parameter objects
- Requiring `GPy>=1.15` and Python `>=3.11` when GPy does
- Exposing DeepGP as a `GPy.Model` subclass with familiar `optimize`, `predict`, and `checkgrad`
- Preferring GPy serialization and parameter arrays over bespoke formats

**Counter-examples**:
- Reimplementing kernels or variational posteriors that already exist in GPy
- Supporting Python or NumPy versions that current GPy has dropped
- Shipping a parallel modelling API that ignores GPy conventions
- Pinning an ancient GPy forever to avoid adapting layer code

**Conflicts**:
- *Conflict*: Stability for old notebooks vs. tracking GPy upgrades
- *Resolution*: Follow GPy's supported range; document breaking changes in release notes and examples
- *Conflict*: Novel deep-GP research features vs. staying inside GPy abstractions
- *Resolution*: Extend via GPy-compatible layers first; only add new primitives when GPy cannot express them
