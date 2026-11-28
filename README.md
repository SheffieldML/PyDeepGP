Deep GP
=======

The Python Implementation of Deep Gaussian Processes

| Branch | Tests | Coverage |
| --- | --- | --- |
| `main` | [![Tests](https://github.com/SheffieldML/PyDeepGP/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/SheffieldML/PyDeepGP/actions/workflows/tests.yml?query=branch%3Amain) | [![codecov](https://codecov.io/gh/SheffieldML/PyDeepGP/branch/main/graph/badge.svg)](https://codecov.io/gh/SheffieldML/PyDeepGP/branch/main) |

Currently implemented models are

* Deep GPs
* Variational Auto-encoded Deep GPs (NumPy MLP back-constraint; see CIP-0001)
* Multi-view / MRD Deep GPs (shared latent layer over observation views; see CIP-0002)

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
encoder and MRD (see `.coveragerc`). Prediction-only SVI helpers remain
omitted until those paths are removed or revived (CIP-0001 Phases 2–4).

Pre-commit (notebooks)
----------------------

Commits strip Jupyter outputs and execution counts via
[nbstripout](https://github.com/kynan/nbstripout) so notebook diffs stay
readable. Install once after cloning::

```bash
python -m pip install pre-commit
pre-commit install
```

Hooks run automatically on ``git commit``. To strip existing notebooks::

```bash
pre-commit run nbstripout --all-files
```

Multi-view / MRD (CIP-0002)
---------------------------

Pass a list of observation matrices and a matching list of output dimensions.
The observation-layer kernels must be a **list of kernels** (one per view)::

    m = deepgp.DeepGP(
        [[D1, D2], Q_shared, Q_top],
        Y=[Y1, Y2],
        kernels=[[kern_v1, kern_v2], kern_top],
        num_inducing=40,
        back_constraint=False,
    )

With ``back_constraint=True``, ``encoder_dims[0]`` is a list of per-view MLP
hidden widths (e.g. ``[[[4], [4]], [3]]``); higher indices feed the usual
``EncoderLayer`` stack for layers above the MRD leaf. See
``examples/example_mrd.py``.

MPI (removed)
-------------

MPI parallel inference has been removed (CIP-0001, CIP-0003). Passing a
non-``None`` ``mpi_comm`` to ``DeepGP``, a layer or an inference method raises
``TypeError``; ``mpi_comm=None`` is still accepted. Use chunked psi-statistics
(``psi_chunk_size``, ``psi_threads``) for single-machine scaling. The old
``SparseGP_MPI`` class name is now an alias of ``SparseGPLayer`` and emits a
``FutureWarning``.

Doubly stochastic inference (new)
---------------------------------

``DeepGP(nDims, Y, X, inference_method='doubly_stochastic', ...)`` returns a
``DoublyStochasticDeepGP`` (Salimbeni and Deisenroth, 2017): a whitened ``q(u)``
per layer, reparameterised samples through hidden layers, and a minibatch
trainer. The likelihood is Gaussian by default; ``likelihood=`` takes a
``deepgp.functional.likelihoods`` object (``Probit``, ``Poisson``, ``StudentT``) with the
top-layer expectation by 20-point Gauss-Hermite quadrature, matching GPc. Natural gradients
need a log-concave likelihood, so ``StudentT`` is refused there. The default collapsed ``DeepGP`` is
unchanged. Pass ``X=None`` for the unsupervised model, with a Gaussian ``q(X)``
(``m.X_mean``, ``m.X_var``) initialised by PCA of ``Y``. Training can give the top
layer natural-gradient steps (``fit_minibatch(..., natgrad_gamma=0.1)``)::

    m = deepgp.DeepGP([1, 2, 1], Y, X, num_inducing=20,
                      inference_method='doubly_stochastic', seed=0)
    m.optimize(max_iters=200)                    # fixed samples: deterministic
    m.fit_minibatch(steps=500, batch_size=50)    # fresh batch and samples each step
    mean, var = m.predict(Xnew)

Unsupervised models take ``back_constraint=True`` (an MLP encoder gives the means of
``q(X)``, as in the collapsed model, initialised near the PCA solution; variances stay free per
point). Multiple views (MRD) pass ``Y`` as a list and ``nDims[0]`` as the list of view widths,
for example ``DeepGP([[6, 5], 2, 3], [A, B], inference_method='doubly_stochastic')``: the
hidden layers are a shared trunk and each view has its own output layer, kernel and noise. With
back constraints the per-view encoders combine by product of experts. ``predict``,
``predict_y`` and ``log_predictive_density`` take ``view=``. MRD supports the Gaussian
likelihood only; see ``examples/mrd_doubly_stochastic.py``.

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
* ``examples/damianou2013/`` — recreations of the AISTATS 2013 experiments
  (toy hierarchy, warped regression, USPS depth/ARD, high-five / MRD),
  including Colab notebooks under ``examples/damianou2013/notebooks/``.
