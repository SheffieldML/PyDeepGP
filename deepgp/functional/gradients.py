# Copyright (c) 2015-2016, the authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""Route gradients with respect to statistics onto layer parameters (pure)."""

import numpy as np


def route_gradients(kern, Z, X, dL_dKmm, dL_dpsi0=None, dL_dpsi1=None,
                    dL_dpsi2=None, dL_dKdiag=None, dL_dKnm=None):
    """Gradients w.r.t. kernel parameters and inducing inputs.

    ``kern`` is the only non-array argument (a GPy kernel), used for its gradient
    methods (which overwrite ``kern.gradient`` as scratch space; the caller
    assigns the returned gradient). ``Z`` and ``X`` are arrays; ``X``
    may be a GPy ``VariationalPosterior`` (uncertain inputs), in which case the
    ``dL_dpsi*`` arguments are used, otherwise ``dL_dKdiag`` and ``dL_dKnm``.

    :returns: ``(kern_gradient, Z_gradient)`` as arrays.
    """
    from GPy.core.parameterization.variational import VariationalPosterior
    if isinstance(X, VariationalPosterior):
        kern.update_gradients_expectations(variational_posterior=X, Z=Z,
                                           dL_dpsi0=dL_dpsi0, dL_dpsi1=dL_dpsi1,
                                           dL_dpsi2=dL_dpsi2)
        kerngrad = kern.gradient.copy()
        kern.update_gradients_full(dL_dKmm, Z, None)
        kerngrad += kern.gradient.copy()
        Zgrad = kern.gradients_Z_expectations(dL_dpsi0, dL_dpsi1, dL_dpsi2,
                                              Z=Z, variational_posterior=X)
        Zgrad = Zgrad + kern.gradients_X(dL_dKmm, Z)
    else:
        kern.update_gradients_diag(dL_dKdiag, X)
        kerngrad = kern.gradient.copy()
        kern.update_gradients_full(dL_dKnm, X, Z)
        kerngrad += kern.gradient
        kern.update_gradients_full(dL_dKmm, Z, None)
        kerngrad = kerngrad + kern.gradient
        Zgrad = kern.gradients_X(dL_dKnm.T, Z, X) + kern.gradients_X(dL_dKmm, Z)
    return np.asarray(kerngrad).copy(), np.asarray(Zgrad)
