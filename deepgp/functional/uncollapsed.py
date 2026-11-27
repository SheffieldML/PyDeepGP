# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""Whitened sparse-GP layer marginals and gradients (pure; CIP-0003 Phase 4).

A layer holds ``P`` independent outputs sharing one kernel and one set of
inducing inputs ``Z``. Each output ``p`` has a whitened variational posterior
``q(v_p) = N(m_p, S_p)`` with ``S_p = L_p L_p^T``, related to the inducing
values by ``u_p = Lm v_p`` where ``Kmm = Lm Lm^T``. At inputs ``X`` the marginals
of ``f_p`` are

    mu_p  = B^T m_p
    var_p = kdiag - colsum(B**2) + colsum((L_p^T B)**2),     B = Lm^{-1} Kmn.

This is the boundary a hidden layer needs (GPy's ``SVGP`` does not expose it):
:func:`whitened_marginals` returns marginals at supplied inputs, and
:func:`whitened_marginals_backward` takes the upstream gradients
``(dL/dmu, dL/dvar)`` and returns gradients for ``q(u)``, the kernel, ``Z`` and
the inputs, so the gradient can continue to the layer below.

Arrays only, except that ``kern`` is a GPy kernel object (see
``deepgp.functional``). Shapes: ``Z (M, Q)``, ``X (N, Q)``, ``q_mu (M, P)``,
``q_chol (P, M, M)`` lower triangular; outputs ``(N, P)``.
"""

import numpy as np
from scipy.linalg import solve_triangular

from .gradients import route_gradients


def whitened_marginals(kern, Z, X, q_mu, q_chol, jitter=1e-6):
    """Marginal mean and variance of each output at ``X``.

    :returns: ``(mu, var, cache)``; pass ``cache`` to the backward function.
    """
    M = Z.shape[0]
    Kmm = kern.K(Z) + jitter * np.eye(M)
    Lm = np.linalg.cholesky(Kmm)
    Knm = kern.K(X, Z)
    B = solve_triangular(Lm, Knm.T, lower=True)
    kdiag = kern.Kdiag(X)
    C = np.einsum('pmk,mn->pkn', q_chol, B)
    mu = B.T @ q_mu
    var = kdiag[:, None] - np.sum(B ** 2, axis=0)[:, None] + np.sum(C ** 2, axis=1).T
    cache = dict(Lm=Lm, B=B, C=C, q_mu=q_mu, q_chol=q_chol)
    return mu, var, cache


def _cholesky_backward(Lm, dLm):
    """Gradient w.r.t. the (symmetric) matrix whose Cholesky factor is ``Lm``."""
    P = np.tril(Lm.T @ dLm)
    P[np.diag_indices_from(P)] *= 0.5
    inv = solve_triangular(Lm, np.eye(Lm.shape[0]), lower=True)  # Lm^{-1}
    dK = inv.T @ P @ inv
    return 0.5 * (dK + dK.T)


def whitened_marginals_backward(kern, Z, X, cache, dL_dmu, dL_dvar):
    """Gradients given upstream ``dL_dmu`` and ``dL_dvar`` (both ``(N, P)``).

    Returns a dict with ``q_mu``, ``q_chol`` (lower triangular), ``kern`` (array,
    the caller assigns it), ``Z`` and ``X``. The ``X`` entry is the gradient to
    pass to the layer below.
    """
    Lm, B, C = cache['Lm'], cache['B'], cache['C']
    q_mu, q_chol = cache['q_mu'], cache['q_chol']
    g_total = dL_dvar.sum(axis=1)

    dq_mu = B @ dL_dmu
    dB = q_mu @ dL_dmu.T - 2. * B * g_total[None, :]
    dq_chol = np.empty_like(q_chol)
    for p in range(q_chol.shape[0]):
        dC = 2. * C[p] * dL_dvar[:, p][None, :]
        dB += q_chol[p] @ dC
        dq_chol[p] = np.tril(B @ dC.T)

    dKmn = solve_triangular(Lm, dB, lower=True, trans='T')      # Lm^{-T} dB
    dLm = -dKmn @ B.T
    dKmm = _cholesky_backward(Lm, dLm)
    dKnm = dKmn.T

    kerngrad, Zgrad = route_gradients(kern, Z, X, dKmm, dL_dKdiag=g_total, dL_dKnm=dKnm)
    Xgrad = kern.gradients_X(dKnm, X, Z) + kern.gradients_X_diag(g_total, X)
    return dict(q_mu=dq_mu, q_chol=dq_chol, kern=kerngrad, Z=Zgrad, X=Xgrad)


def whitened_kl(q_mu, q_chol):
    """``sum_p KL(N(m_p, L_p L_p^T) || N(0, I))`` and its gradients.

    :returns: ``(kl, dq_mu, dq_chol)``.
    """
    M = q_mu.shape[0]
    diag = np.stack([np.diag(L) for L in q_chol])
    kl = 0.5 * (np.sum(q_chol ** 2) + np.sum(q_mu ** 2) - M * q_chol.shape[0]
                - 2. * np.sum(np.log(np.abs(diag))))
    dq_chol = np.array(q_chol, dtype=float)
    for p in range(q_chol.shape[0]):
        dq_chol[p][np.diag_indices(M)] -= 1. / diag[p]
        dq_chol[p] = np.tril(dq_chol[p])
    return kl, q_mu.copy(), dq_chol


def reparam_sample(mu, var, eps):
    """``f = mu + sqrt(var) * eps`` with caller-supplied standard normal ``eps``."""
    return mu + np.sqrt(var) * eps


def reparam_sample_backward(var, eps, dL_df):
    """Gradients ``(dL_dmu, dL_dvar)`` of :func:`reparam_sample`."""
    return dL_df, dL_df * eps / (2. * np.sqrt(var))
