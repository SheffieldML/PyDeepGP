Deep GP
=======

The Python Implementation of Deep Gaussian Processes

Currently implemented models are

* Deep GPs
* Variational Auto-encoded Deep GPs

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

Back-constraint encoder tests are skipped unless Theano is installed.

Coverage is gated on the maintained serial DeepGP path (see `.coveragerc`).
MPI parallel inference, MRD, Theano encoder, and prediction-only SVI helpers
are omitted from the CI percentage until those paths are revived.
