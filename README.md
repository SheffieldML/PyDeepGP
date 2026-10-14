Deep GP
=======

The Python Implementation of Deep Gaussian Processes

Currently implemented models are

* Deep GPs
* Variational Auto-encoded Deep GPs

Testing
-------

GitHub Actions runs the unit tests on pull requests and pushes to `main`.

To run the suite locally:

```bash
python -m pip install "numpy>=1.7" "scipy>=0.12" "GPy>=1.0" matplotlib h5py tables
python -m pip install -e . --no-deps
python -m unittest deepgp.testing.model_tests_basic -v
```

Or with [tox](https://tox.readthedocs.io/):

```bash
tox
```
