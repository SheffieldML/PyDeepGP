"""Doubly stochastic bound: GPy SVGP agreement and finite-difference gradients."""
import unittest

import numpy as np
import GPy
from GPy.util import choleskies

from deepgp.functional import doubly_stochastic_bound
from deepgp.testing.test_uncollapsed import _fd, _fd_kern


def _layer(rng, Q, P, M, residual=False, ls=1.):
    kern = GPy.kern.RBF(Q, ARD=True, lengthscale=[ls] * Q, variance=1.2)
    return dict(kern=kern, Z=rng.randn(M, Q), q_mu=0.5 * rng.randn(M, P),
                q_chol=np.stack([np.tril(0.2 * rng.randn(M, M)) + np.eye(M) * (0.6 + rng.rand(M)) for _ in range(P)]),
                residual=residual)


class SingleLayerVsGPyTest(unittest.TestCase):
    def test_matches_gpy_svgp(self):
        rng = np.random.RandomState(0)
        N, Q, M, P = 12, 2, 5, 1
        X, Y = rng.randn(N, Q), rng.randn(N, P)
        layer = _layer(rng, Q, P, M)
        noise = 0.3
        bound, _ = doubly_stochastic_bound([layer], X, Y, noise, eps=[[]])

        Lm = np.linalg.cholesky(layer['kern'].K(layer['Z']) + 1e-6 * np.eye(M))
        m = GPy.core.SVGP(X, Y, layer['Z'].copy(), layer['kern'].copy(),
                          GPy.likelihoods.Gaussian(variance=noise))
        m.q_u_mean[:] = Lm @ layer['q_mu']
        chol = np.stack([Lm @ layer['q_chol'][p] for p in range(P)])  # (P, M, M)
        m.q_u_chol[:] = choleskies.triang_to_flat(chol)
        self.assertAlmostEqual(bound, float(m.log_likelihood()), delta=1e-4 * (1 + abs(bound)))


class StackGradientTest(unittest.TestCase):
    def _build(self, residual):
        rng = np.random.RandomState(2)
        N, Q = 6, 2
        X, Y = rng.randn(N, Q), rng.randn(N, 1)
        layers = [_layer(rng, Q, 2, 4, residual=residual), _layer(rng, 2, 2, 3, residual=residual),
                  _layer(rng, 2, 1, 4)]
        eps = [[rng.randn(N, 2), rng.randn(N, 2)] for _ in range(2)]
        return layers, X, Y, eps

    def _check(self, residual):
        layers, X, Y, eps = self._build(residual)
        noise = [0.4]
        F = lambda: doubly_stochastic_bound(layers, X.copy(), Y, noise[0], eps, scale=2.5, kl_scale=0.7)[0]
        _, g = doubly_stochastic_bound(layers, X.copy(), Y, noise[0], eps, scale=2.5, kl_scale=0.7)
        tol = dict(rtol=1e-5, atol=1e-6)
        for l, layer in enumerate(layers):
            P, M = layer['q_mu'].shape[1], layer['q_mu'].shape[0]
            mask = np.broadcast_to(np.tril(np.ones((M, M), dtype=bool)), layer['q_chol'].shape)
            np.testing.assert_allclose(g['layers'][l]['q_mu'], _fd(F, layer['q_mu']), **tol)
            np.testing.assert_allclose(g['layers'][l]['q_chol'], _fd(F, layer['q_chol'], mask=mask), **tol)
            np.testing.assert_allclose(g['layers'][l]['Z'], _fd(F, layer['Z']), **tol)
            np.testing.assert_allclose(g['layers'][l]['kern'], _fd_kern(F, layer['kern']), **tol)
        np.testing.assert_allclose(g['X'], _fd(F, X), **tol)

        def Fn():
            return doubly_stochastic_bound(layers, X.copy(), Y, noise[0], eps, scale=2.5, kl_scale=0.7)[0]
        h = 1e-6
        v = noise[0]
        noise[0] = v + h; fp = Fn()
        noise[0] = v - h; fm = Fn()
        self.assertAlmostEqual(g['noise_var'], (fp - fm) / (2 * h), places=4)

    def test_gradients_plain(self):
        self._check(residual=False)

    def test_gradients_with_residual_mean(self):
        self._check(residual=True)

    def test_validation(self):
        layers, X, Y, eps = self._build(False)
        with self.assertRaises(ValueError):
            doubly_stochastic_bound(layers, X, Y, 0.4, [[eps[0][0]]])       # too few eps
        with self.assertRaises(ValueError):
            doubly_stochastic_bound(layers, X, np.hstack([Y, Y]), 0.4, eps)  # wrong Y width
        bad = [dict(layers[0], residual=True)] + layers[1:]                  # Q=2 -> P=2 fine
        bad[0] = dict(bad[0], q_mu=bad[0]['q_mu'][:, :1], q_chol=bad[0]['q_chol'][:1])
        with self.assertRaises(ValueError):
            doubly_stochastic_bound(bad, X, Y, 0.4, eps)


class LatentInputTest(unittest.TestCase):
    def test_q_x_gradients(self):
        rng = np.random.RandomState(4)
        N, Q = 6, 2
        X, Y = rng.randn(N, Q), rng.randn(N, 3)
        x_var = 0.2 + rng.rand(N, Q)
        layers = [_layer(rng, Q, 2, 4, residual=True), _layer(rng, 2, 3, 4)]
        eps = [[rng.randn(N, 2)] for _ in range(2)]
        eps_x = [rng.randn(N, Q) for _ in range(2)]
        kw = dict(scale=2., eps_x=eps_x)
        F = lambda: doubly_stochastic_bound(layers, X.copy(), Y, 0.4, eps, x_var=x_var.copy(), **kw)[0]
        _, g = doubly_stochastic_bound(layers, X.copy(), Y, 0.4, eps, x_var=x_var.copy(), **kw)
        np.testing.assert_allclose(g['X'], _fd(F, X), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(g['x_var'], _fd(F, x_var), rtol=1e-5, atol=1e-6)

    def test_zero_variance_reduces_to_deterministic_input(self):
        rng = np.random.RandomState(5)
        X, Y = rng.randn(5, 2), rng.randn(5, 1)
        layers = [_layer(rng, 2, 1, 4)]
        b0, _ = doubly_stochastic_bound(layers, X, Y, 0.3, [[]])
        v = np.full_like(X, 1e-14)
        b1, _ = doubly_stochastic_bound(layers, X, Y, 0.3, [[]], x_var=v, eps_x=[rng.randn(5, 2)])
        kl_x = 0.5 * np.sum(X ** 2 + v - 1. - np.log(v))
        self.assertAlmostEqual(b1, b0 - kl_x, places=6)

    def test_requires_eps_x(self):
        rng = np.random.RandomState(6)
        X, Y = rng.randn(5, 2), rng.randn(5, 1)
        with self.assertRaises(ValueError):
            doubly_stochastic_bound([_layer(rng, 2, 1, 4)], X, Y, 0.3, [[]], x_var=np.ones_like(X))


if __name__ == '__main__':
    unittest.main()
