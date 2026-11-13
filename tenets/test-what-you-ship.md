---
conflicts_with: []
created: '2026-11-13'
id: test-what-you-ship
last_reviewed: '2026-11-13'
last_updated: '2026-11-13'
review_frequency: Annual
status: Active
tags:
- tenet
- testing
- coverage
- ci
title: Test What You Ship
version: '1.0'
---

# Test What You Ship

## Tenet

**Title**: Test What You Ship

**Description**: Behaviour that users are told to rely on must be exercised in automated tests on CI. Coverage gates should reflect the maintained surface, not inflate scores by ignoring breakage in advertised APIs—or by demanding coverage of deferred modules. Golden-file likelihood checks, gradients, prediction shapes, and clear failure modes for unsupported entry points all count.

**Quote**: *"CI is the contract with the user"*

**Examples**:
- Unittest discovery covering basic likelihood/gradient checks plus API and util tests
- CI matrix on Python versions supported by current GPy
- Coverage fail-under applied to the maintained serial path in `.coveragerc`
- Tests that `gen_pred_*` raises `NotImplementedError` rather than a raw import failure

**Counter-examples**:
- Advertising prediction helpers with no test that they run
- Running CI only on Python versions GPy no longer installs on
- Setting a coverage gate so low that core regressions land unnoticed
- Counting deferred MPI/MRD files against contributors who are fixing the spine

**Conflicts**:
- *Conflict*: Fast CI vs. thorough numerical tests
- *Resolution*: Keep unit tests small and deterministic; park long optimisations in examples or optional jobs
- *Conflict*: Testing optional Theano encoders vs. a lean default install
- *Resolution*: SkipUnless optional deps; do not fail the default matrix when they are absent
