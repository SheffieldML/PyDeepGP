Deep GP
=======

The Python Implementation of Deep Gaussian Processes

Currently implemented models are

* Deep GPs
* Variational Auto-encoded Deep GPs

Testing
-------

GitHub Actions runs the unit tests on pull requests and pushes to `master`.

To run the suite locally:

```bash
python -m pip install -e .
python -m pip install matplotlib h5py tables
python -m unittest deepgp.testing.model_tests_basic -v
```

Or with [tox](https://tox.readthedocs.io/):

```bash
tox
```
