# Copyright (c) 2015-2016, the authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""Pure functions for the variational distribution q(X) of a layer's inputs."""

import numpy as np


def qx_gradients(kern, Z, X, dL_dpsi0, dL_dpsi1, dL_dpsi2):
    """Gradients of the bound w.r.t. the mean and variance of q(X).

    :param kern: GPy kernel (the one non-array argument, see ``route_gradients``).
    :param X: GPy ``VariationalPosterior`` supplying the values.
    :returns: ``(dL_dmean, dL_dvar)`` arrays.
    """
    return kern.gradients_qX_expectations(variational_posterior=X, Z=Z,
                                          dL_dpsi0=dL_dpsi0, dL_dpsi1=dL_dpsi1,
                                          dL_dpsi2=dL_dpsi2)


def aggregate_qx(view_means, view_variances):
    """Product-of-experts combination of per-view Gaussians (MRD).

    Precisions add and the mean is the precision-weighted mean.

    :returns: ``(mean, variance, prec_denom)`` with ``prec_denom`` the summed precision.
    """
    prec_denom = np.zeros_like(view_variances[0])
    mean = np.zeros_like(view_means[0])
    for m, v in zip(view_means, view_variances):
        prec_denom = prec_denom + 1. / v
        mean = mean + m / v
    return mean / prec_denom, 1. / prec_denom, prec_denom


def disaggregate_qx_gradients(view_means, view_variances, prec_denom,
                              mean, dL_dmean, dL_dvar):
    """Back-propagate gradients on the aggregate q(X) to each view's q(X).

    Inverse of :func:`aggregate_qx` in the chain-rule sense, as used by the MRD
    layer for the per-view encoders.

    :returns: list of ``(dL_dmean_v, dL_dvar_v)``, one per view.
    """
    out = []
    for m, v in zip(view_means, view_variances):
        c = 1. / (prec_denom * v)
        out.append((c * dL_dmean,
                    np.square(c) * dL_dvar + c * (mean - m) / v * dL_dmean))
    return out
