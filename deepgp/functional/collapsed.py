# Copyright (c) 2015-2016, the authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""Collapsed variational bound of a sparse GP layer, as a pure function."""

import numpy as np
from GPy.util.linalg import jitchol, backsub_both_sides, tdot, dtrtrs

log_2_pi = np.log(2 * np.pi)


def collapsed_bound(Kmm, psi0, psi1, psi2, Y_mean, beta, Y_var=None):
    """Collapsed bound with Gaussian noise, for certain or uncertain outputs.

    All arguments are arrays (``beta`` a positive float). Statistics are
    *unscaled* by the precision.

    :param Kmm: M x M covariance of the inducing outputs, jitter included.
    :param psi0: (N,) or scalar-sum of E[k(x, x)].
    :param psi1: N x M, E[k(x, Z)].
    :param psi2: M x M, sum_n E[k(Z, x_n) k(x_n, Z)]; ``None`` for deterministic
        inputs, in which case it is ``psi1.T @ psi1``.
    :param Y_mean: N x D output (mean when ``Y_var`` is given).
    :param beta: noise precision.
    :param Y_var: optional N x D output variances (uncertain outputs).
    :returns: dict with ``logL`` (shape (1,)), ``dL_dKmm``, ``dL_dpsi0`` (N,),
        ``dL_dpsi1``, ``dL_dpsi2`` (``None`` when ``psi2`` is ``None``, the
        gradient having been folded into ``dL_dpsi1``), ``dL_dthetaL``,
        ``woodbury_vector``, ``woodbury_inv``, ``cov``, ``Lm`` and, for uncertain
        outputs, ``dL_dYmean`` (N x D) and ``dL_dYvar`` ((N,): the bound depends on
        the variances only through their row sums).
    """
    num_data, output_dim = Y_mean.shape
    input_dim = Kmm.shape[0]
    deterministic_inputs = psi2 is None
    beta = float(beta)

    psi0_sum = np.sum(psi0) * beta
    psi1b = psi1 * beta
    psi2b = (psi1.T @ psi1 if deterministic_inputs else psi2) * beta
    psi1Y = np.dot(Y_mean.T, psi1b)  # D x M
    if Y_var is not None:
        Shalf = np.sqrt(Y_var.sum(axis=1))
        psi1S = Shalf[:, None] * psi1b
        YRY = (np.square(Y_mean).sum() + Y_var.sum()) * beta
    else:
        Shalf = psi1S = None
        YRY = np.sum(np.square(Y_mean)) * beta

    logL_R = -num_data * np.log(beta)

    Lm = jitchol(Kmm)
    LmInvPsi2LmInvT = backsub_both_sides(Lm, psi2b, 'right')
    Lambda = np.eye(input_dim) + LmInvPsi2LmInvT
    LL = jitchol(Lambda)
    LmLL = Lm.dot(LL)

    logdet_L = 2. * np.sum(np.log(np.diag(LL)))
    b = dtrtrs(LmLL, psi1Y.T)[0].T
    bbt = np.square(b).sum()
    v = dtrtrs(LmLL, b.T, trans=1)[0].T
    LLinvPsi1TYYTPsi1LLinvT = tdot(b.T)

    if psi1S is not None:
        psi1SLLinv = dtrtrs(LmLL, psi1S.T)[0].T
        bbt += np.square(psi1SLLinv).sum()
        LLinvPsi1TYYTPsi1LLinvT += tdot(psi1SLLinv.T)
        psi1SP = dtrtrs(LmLL, psi1SLLinv.T, trans=1)[0].T
    tmp = -backsub_both_sides(LL, LLinvPsi1TYYTPsi1LLinvT + output_dim * np.eye(input_dim))
    dL_dpsi2R = backsub_both_sides(Lm, tmp + output_dim * np.eye(input_dim)) / 2

    logL = -(output_dim * (num_data * log_2_pi + logL_R + psi0_sum - np.trace(LmInvPsi2LmInvT))
             + YRY - bbt) / 2. - output_dim * logdet_L / 2.

    dL_dKmm = dL_dpsi2R - output_dim * backsub_both_sides(Lm, LmInvPsi2LmInvT) / 2

    LLInvLmT = dtrtrs(LL, Lm.T)[0]
    cov = tdot(LLInvLmT.T)
    wd_inv = backsub_both_sides(
        Lm, np.eye(input_dim) - backsub_both_sides(LL, np.identity(input_dim), transpose='left'),
        transpose='left')

    dL_dthetaL = ((YRY * beta + beta * output_dim * psi0_sum - num_data * output_dim * beta) / 2.
                  - beta * (dL_dpsi2R * psi2b).sum()
                  - beta * np.trace(LLinvPsi1TYYTPsi1LLinvT))

    dL_dpsi0 = -output_dim * (beta * np.ones((num_data,))) / 2.
    if Y_var is not None:
        dL_dpsi1 = beta * (np.dot(Y_mean, v) + Shalf[:, None] * psi1SP)
    else:
        dL_dpsi1 = beta * np.dot(Y_mean, v)
    if deterministic_inputs:
        dL_dpsi1 += np.dot(psi1b, dL_dpsi2R) * 2.
        dL_dpsi2 = None
    else:
        dL_dpsi2 = beta * dL_dpsi2R

    out = {'logL': np.asarray(logL).reshape(1), 'dL_dKmm': dL_dKmm,
           'dL_dpsi0': dL_dpsi0, 'dL_dpsi1': dL_dpsi1, 'dL_dpsi2': dL_dpsi2,
           'dL_dthetaL': dL_dthetaL, 'woodbury_vector': v.T,
           'woodbury_inv': wd_inv, 'cov': cov, 'Lm': Lm}
    if Y_var is not None:
        psi1LmiLLi = dtrtrs(LmLL, psi1b.T)[0].T
        out['dL_dYmean'] = -Y_mean * beta + psi1LmiLLi.dot(b.T)
        out['dL_dYvar'] = beta / -2. + np.square(psi1LmiLLi).sum(axis=1) / 2
    return out
