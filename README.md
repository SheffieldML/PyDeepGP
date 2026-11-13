Deep GP
=======

The Python Implementation of Deep Gaussian Processes

Currently implemented models are

* Deep GPs
* Variational Auto-encoded Deep GPs (NumPy MLP back-constraint; see CIP-0001)

Testing
-------

GitHub Actions runs the unit tests (with coverage) on pull requests and pushes to `main`.

Requires **Python >=3.11** when using GPy 1.15+.

To run the suite locally:

```bash
python -m pip install -e ".[test]"
# or, if editable extras are unreliable:
python -m pip install "numpy>=1.7" "scipy>=0.12" "GPy>=1.15" matplotlib h5py tables "coverage>=7.0"
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
