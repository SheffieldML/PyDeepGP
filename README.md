Deep GP
=======

The Python Implementation of Deep Gaussian Processes

| Branch | Tests | Coverage |
| --- | --- | --- |
| `main` | [![Tests](https://github.com/SheffieldML/PyDeepGP/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/SheffieldML/PyDeepGP/actions/workflows/tests.yml?query=branch%3Amain) | [![codecov](https://codecov.io/gh/SheffieldML/PyDeepGP/branch/main/graph/badge.svg)](https://codecov.io/gh/SheffieldML/PyDeepGP/branch/main) |

Currently implemented models are

* Deep GPs
* Variational Auto-encoded Deep GPs (NumPy MLP back-constraint; see CIP-0001)

Testing
-------

GitHub Actions runs the unit tests (with coverage) on pull requests and pushes to `main`.
The Ubuntu / Python 3.12 job uploads ``coverage.xml`` to [Codecov](https://codecov.io).

Requires **Python >=3.11** when using GPy 1.15+.

To run the suite locally:

```bash
python -m pip install -e ".[test]"
# or, if editable extras are unreliable:
python -m pip install "numpy>=1.7" "scipy>=0.12" "GPy>=1.15" matplotlib h5py tables "coverage>=7.0"
# For psi_chunk_size / psi_threads (CIP-0001 Phase 3) until the next GPy release:
# python -m pip install "git+https://github.com/SheffieldML/GPy.git@main"
python -m pip install -e . --no-deps

python -m coverage run -m unittest discover -s deepgp/testing -p 'test*.py' -v
python -m coverage report
```

Or with [tox](https://tox.readthedocs.io/):

```bash
tox
```

Coverage is gated on the maintained serial DeepGP path, including the NumPy
encoder (see `.coveragerc`). MPI parallel inference, MRD, and prediction-only
SVI helpers remain omitted until those paths are revived or removed
(CIP-0001 Phases 2–4).

MPI (deprecated)
----------------

Passing ``mpi_comm`` to ``DeepGP`` emits a ``FutureWarning`` (CIP-0001). Use
serial models for now; chunked psi-statistics (Phase 3) will replace the
single-machine role of MPI. ``mpi_comm`` with ``back_constraint=False`` raises
``ValueError`` (that configuration produced silently wrong results).

An optional GitHub Actions job runs a 2-rank parity check while MPI remains.

Psi chunking (optional)
-----------------------

Large-N models can bound memory and use multiple cores via GPy's chunked RBF /
linear psi-statistics (CIP-0001 Phase 3)::

    m = deepgp.DeepGP(..., psi_chunk_size=500, psi_threads=4)

Defaults (``psi_chunk_size=None``, ``psi_threads=1``) preserve previous
behaviour. Threading runs inside GPy's pure psi helpers; PyDeepGP itself stays
serial at the kernel API.

Examples
--------

* ``examples/`` — short supervised / unsupervised demos (oil, CMU walk/jog).
* ``examples/damianou2013/`` — recreations of the feasible AISTATS 2013
  experiments (toy hierarchy, warped regression, USPS depth/ARD). The CMU
  high-five shared-view experiment still needs an MRD revival; see that
  directory's README.
