"""Whitened marginals core: agreement with dense algebra and finite differences."""
import unittest

import numpy as np
import GPy

from deepgp.functional import (whitened_marginals, whitened_marginals_backward, whitened_kl,
                               reparam_sample, reparam_sample_backward)


def _setup(seed=0, N=7, M=4, Q=2, P=3):
    rng = np.random.RandomState(seed)
    kern = GPy.kern.RBF(Q, ARD=True, lengthscale=[0.8, 1.3], variance=1.4)
    Z, X = rng.randn(M, Q), rng.randn(N, Q)
    q_mu = rng.randn(M, P)
    q_chol = np.stack([np.tril(0.3 * rng.randn(M, M)) + np.eye(M) * (0.5 + rng.rand(M)) for _ in range(P)])
    return kern, Z, X, q_mu, q_chol, rng


def _fd(f, x, eps=1e-6, mask=None):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    for _ in it:
        i = it.multi_index
        if mask is not None and not mask[i]:
            continue
        old = x[i]
        x[i] = old + eps; fp = f()
        x[i] = old - eps; fm = f()
        x[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g


def _fd_kern(f, kern, eps=1e-6):
    """Finite differences over kernel parameter values (set via the Parameterized API)."""
    base = kern.param_array.copy()
    g = np.zeros_like(base)
    for i in range(base.size):
        for sign in (1, -1):
            v = base.copy(); v[i] += sign * eps
            kern[:] = v
            g[i] += sign * f()
    kern[:] = base
    return g / (2 * eps)


class WhitenedMarginalsTest(unittest.TestCase):
    def test_matches_dense_unwhitened_algebra(self):
        kern, Z, X, m, L, _ = _setup()
        jit = 1e-6
        mu, var, _ = whitened_marginals(kern, Z, X, m, L, jitter=jit)
        Kmm = kern.K(Z) + jit * np.eye(len(Z))
        Lm = np.linalg.cholesky(Kmm)
        Knm, Kinv = kern.K(X, Z), np.linalg.inv(Kmm)
        for p in range(m.shape[1]):
            mu_u, S_u = Lm @ m[:, p], Lm @ L[p] @ L[p].T @ Lm.T      # unwhitened q(u)
            np.testing.assert_allclose(mu[:, p], Knm @ Kinv @ mu_u, rtol=1e-8, atol=1e-8)
            cov = kern.K(X) - Knm @ Kinv @ Knm.T + Knm @ Kinv @ S_u @ Kinv @ Knm.T
            np.testing.assert_allclose(var[:, p], np.diag(cov), rtol=1e-6, atol=1e-8)

    def test_zero_variance_posterior_is_prior_mean_free(self):
        kern, Z, X, m, L, _ = _setup()
        mu, var, _ = whitened_marginals(kern, Z, X, np.zeros_like(m), np.stack([np.eye(4)] * 3))
        np.testing.assert_allclose(mu, 0.)
        np.testing.assert_allclose(var, kern.Kdiag(X)[:, None] * np.ones((1, 3)), rtol=1e-6)

    def test_backward_matches_finite_differences(self):
        kern, Z, X, m, L, rng = _setup()
        a, b = rng.randn(*(X.shape[0], m.shape[1])), rng.randn(X.shape[0], m.shape[1])

        def F():
            mu, var, _ = whitened_marginals(kern, Z.copy(), X.copy(), m, L)  # copies: GPy caches on array identity
            return np.sum(a * mu + b * var)

        _, _, cache = whitened_marginals(kern, Z, X, m, L)
        g = whitened_marginals_backward(kern, Z, X, cache, a, b)
        np.testing.assert_allclose(g['q_mu'], _fd(F, m), rtol=1e-5, atol=1e-6)
        low = np.tril(np.ones((4, 4), dtype=bool))
        mask = np.broadcast_to(low, L.shape)
        np.testing.assert_allclose(g['q_chol'], _fd(F, L, mask=mask), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(g['Z'], _fd(F, Z), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(g['X'], _fd(F, X), rtol=1e-5, atol=1e-6)
        kg = _fd_kern(F, kern)
        np.testing.assert_allclose(g['kern'], kg, rtol=1e-5, atol=1e-6)

    def test_kl_matches_dense_and_finite_differences(self):
        _, _, _, m, L, _ = _setup()
        kl, dm, dL = whitened_kl(m, L)
        dense = 0.
        for p in range(m.shape[1]):
            S = L[p] @ L[p].T
            dense += 0.5 * (np.trace(S) + m[:, p] @ m[:, p] - len(m) - np.linalg.slogdet(S)[1])
        self.assertAlmostEqual(kl, dense, places=10)
        np.testing.assert_allclose(dm, _fd(lambda: whitened_kl(m, L)[0], m), rtol=1e-6, atol=1e-7)
        mask = np.broadcast_to(np.tril(np.ones((4, 4), dtype=bool)), L.shape)
        np.testing.assert_allclose(dL, _fd(lambda: whitened_kl(m, L)[0], L, mask=mask), rtol=1e-5, atol=1e-6)


class TwoLayerStackTest(unittest.TestCase):
    """Upstream gradients flow through a reparameterised sample into the layer below."""

    def test_gradient_through_two_layers_with_frozen_eps(self):
        k1, Z1, X, m1, L1, rng = _setup(seed=1, P=2)          # layer 1: 2 outputs
        k2 = GPy.kern.RBF(2, ARD=True, lengthscale=[1., 1.5])
        Z2 = rng.randn(4, 2)
        m2 = rng.randn(4, 1)
        L2 = np.stack([np.tril(0.2 * rng.randn(4, 4)) + np.eye(4)])
        eps = rng.randn(X.shape[0], 2)
        a, b = rng.randn(X.shape[0], 1), rng.randn(X.shape[0], 1)

        def forward():
            mu1, v1, c1 = whitened_marginals(k1, Z1.copy(), X.copy(), m1, L1)
            f1 = reparam_sample(mu1, v1, eps)
            mu2, v2, c2 = whitened_marginals(k2, Z2.copy(), f1, m2, L2)
            return np.sum(a * mu2 + b * v2), (v1, c1, c2, f1)

        F = lambda: forward()[0]
        _, (v1, c1, c2, f1) = forward()
        g2 = whitened_marginals_backward(k2, Z2, f1, c2, a, b)
        dmu1, dv1 = reparam_sample_backward(v1, eps, g2['X'])
        g1 = whitened_marginals_backward(k1, Z1, X, c1, dmu1, dv1)
        np.testing.assert_allclose(g1['X'], _fd(F, X), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(g1['q_mu'], _fd(F, m1), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(g1['Z'], _fd(F, Z1), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(g1['kern'], _fd_kern(F, k1), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(g2['kern'], _fd_kern(F, k2), rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
