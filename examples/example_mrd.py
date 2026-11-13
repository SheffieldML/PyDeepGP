#!/usr/bin/env python
"""Multi-view DeepGP (MRD) demo — CIP-0002.

Two correlated observation views share a parent latent layer. This is the
architecture used for the Damianou & Lawrence (2013) high-five experiment;
see also ``examples/damianou2013/highfive_shared.py``.
"""

from __future__ import annotations

import numpy as np
import GPy
import deepgp

np.random.seed(42)

# Synthetic shared-latent data (replace with two MOCAP subjects for high-five)
n, q_true, d = 60, 3, 20
X_true = np.random.randn(n, q_true)
Y1 = X_true @ np.random.randn(q_true, d) + 0.05 * np.random.randn(n, d)
Y2 = X_true @ np.random.randn(q_true, d) + 0.05 * np.random.randn(n, d)
Y1 = (Y1 - Y1.mean(0)) / (Y1.std() + 1e-12)
Y2 = (Y2 - Y2.mean(0)) / (Y2.std() + 1e-12)

Q_shared, Q_top = 5, 3
num_inducing = 25

kern_views = [
    GPy.kern.RBF(Q_shared, ARD=True) + GPy.kern.Bias(Q_shared),
    GPy.kern.RBF(Q_shared, ARD=True) + GPy.kern.Bias(Q_shared),
]
kern_top = GPy.kern.RBF(Q_top, ARD=True) + GPy.kern.Bias(Q_top)

m = deepgp.DeepGP(
    [[Y1.shape[1], Y2.shape[1]], Q_shared, Q_top],
    [Y1, Y2],
    kernels=[kern_views, kern_top],
    num_inducing=num_inducing,
    back_constraint=False,
)

from GPy.core.parameterization.variational import NormalPosterior, VariationalPosterior


def _out_var(y):
    if isinstance(y, (NormalPosterior, VariationalPosterior)):
        return float(np.asarray(y.mean).var())
    return float(np.asarray(y).var())


for layer in m.layers:
    if hasattr(layer, "views"):
        for view in layer.views:
            view.Gaussian_noise.variance = _out_var(view.Y) * 0.01
            view.Gaussian_noise.variance.fix()
    else:
        layer.Gaussian_noise.variance = _out_var(layer.Y) * 0.01
        layer.Gaussian_noise.variance.fix()

m.optimize(max_iters=200, messages=True)
for layer in m.layers:
    if hasattr(layer, "views"):
        for view in layer.views:
            view.Gaussian_noise.variance.unfix()
    else:
        layer.Gaussian_noise.variance.unfix()
m.optimize(max_iters=500, messages=True)

print(f"MRD DeepGP variational bound: {float(m.log_likelihood()):.3f}")
for i, view in enumerate(m.layers[0].views):
    ls = view.kern.rbf.lengthscale.values if hasattr(view.kern, "rbf") else view.kern.lengthscale.values
    print(f"View {i} ARD (1/lengthscale): {1.0 / np.maximum(ls, 1e-12)}")
