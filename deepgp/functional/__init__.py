# Copyright (c) 2015-2016, the authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""Pure (array-in / array-out) layer mathematics (CIP-0003 Phase 2).

Public but provisional: names and signatures may change before 1.0.

Layer interface
---------------
A layer of a deep GP maps an input distribution q(X) (or fixed X) to marginals
over the layer output, and receives gradients from the layer above. In this
package a layer is split into

* a *parameter shell* (``deepgp.layers``): GPy ``Parameterized`` objects holding
  the kernel, inducing inputs, noise and q(X);
* *pure functions* (this package) that take arrays and return arrays.

The collapsed (VarDTC) family is implemented by :func:`collapsed_bound`
(statistics -> bound, posterior and gradients w.r.t. the statistics) and
:func:`route_gradients` (gradients w.r.t. statistics -> gradients w.r.t. kernel
parameters, inducing inputs and q(X)). The second function receives the GPy
kernel *object* as its single non-array argument, since GPy kernels are not yet
pure functions (GPy CIP-000F Stage 2). It never touches ``Param`` objects,
observers or communicators and returns plain arrays; the shell assigns them.

The uncollapsed family (CIP-0003 Phase 4) will implement the same two roles:
forward marginals at supplied inputs, and gradient routing from upstream
gradients to parameters.
"""

from .collapsed import collapsed_bound
from .gradients import route_gradients

__all__ = ["collapsed_bound", "route_gradients"]
