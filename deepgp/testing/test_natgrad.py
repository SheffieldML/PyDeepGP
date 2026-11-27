"""Natural-gradient step for the top layer's whitened q(u)."""
import unittest

import numpy as np
import GPy

from deepgp.functional import (doubly_stochastic_bound, whitened_natgrad_step,
                               whitened_marginals, whitened_expectation_grads)
from deepgp.models import DoublyStochasticDeepGP


def _one_layer(seed=0, M=6, N=30):
    rng = np.random.RandomState(seed)
    X = rng.randn(N, 2)
    Y = np.sin(X[:, :1]) + 0.1 * rng.randn(N, 1)
    kern = GPy.kern.RBF(2, lengthscale=1.2, variance=0.8)
    Z = rng.randn(M, 2)
    q_mu = 0.3 * rng.randn(M, 1)
    q_chol = np.tril(0.2 * rng.randn(1, M, M)) + np.eye(M)[None] * 0.7
    layer = dict(kern=kern, Z=Z, q_mu=q_mu, q_chol=q_chol, residual=False)
    return X, Y, layer


class TestNatGrad(unittest.TestCase):
    def test_gamma_one_reaches_optimum_gaussian(self):
        X, Y, layer = _one_layer()
        noise = 0.05
        _, g = doubly_stochastic_bound([layer], X, Y, noise, [[]])
        mu, chol = whitened_natgrad_step(layer['q_mu'], layer['q_chol'],
                                         g['natgrad']['g_m'], g['natgrad']['g_S'], 1.)
        layer2 = dict(layer, q_mu=mu, q_chol=chol)
        b1, g1 = doubly_stochastic_bound([layer2], X, Y, noise, [[]])
        np.testing.assert_allclose(g1['layers'][0]['q_mu'], 0., atol=1e-6)
        np.testing.assert_allclose(g1['layers'][0]['q_chol'], 0., atol=1e-6)
        b0, _ = doubly_stochastic_bound([layer], X, Y, noise, [[]])
        self.assertGreater(b1, b0)

    def test_matches_analytic_optimum(self):
        # The optimal Gaussian q(u) under the whitened prior has precision I + B B^T / noise.
        X, Y, layer = _one_layer(seed=1)
        noise = 0.1
        _, g = doubly_stochastic_bound([layer], X, Y, noise, [[]])
        mu, chol = whitened_natgrad_step(layer['q_mu'], layer['q_chol'],
                                         g['natgrad']['g_m'], g['natgrad']['g_S'], 1.)
        _, _, cache = whitened_marginals(layer['kern'], layer['Z'], X, layer['q_mu'], layer['q_chol'])
        B = cache['B']
        S = np.linalg.inv(np.eye(B.shape[0]) + B @ B.T / noise)
        np.testing.assert_allclose(chol[0] @ chol[0].T, S, atol=1e-6)
        np.testing.assert_allclose(mu[:, 0], S @ B @ Y[:, 0] / noise, atol=1e-6)

    def test_small_step_increases_bound(self):
        X, Y, layer = _one_layer(seed=2)
        b0, g = doubly_stochastic_bound([layer], X, Y, 0.1, [[]])
        mu, chol = whitened_natgrad_step(layer['q_mu'], layer['q_chol'],
                                         g['natgrad']['g_m'], g['natgrad']['g_S'], 0.1)
        b1, _ = doubly_stochastic_bound([dict(layer, q_mu=mu, q_chol=chol)], X, Y, 0.1, [[]])
        self.assertGreater(b1, b0)

    def test_expectation_grads_consistent_with_chol_grad(self):
        X, Y, layer = _one_layer(seed=3)
        _, g = doubly_stochastic_bound([layer], X, Y, 0.1, [[]], kl_scale=0.)
        gS = g['natgrad']['g_S'][0]
        L = layer['q_chol'][0]
        np.testing.assert_allclose(np.tril(2. * gS @ L), g['layers'][0]['q_chol'][0], atol=1e-8)

    def test_model_hybrid_training_beats_adam_start(self):
        rng = np.random.RandomState(0)
        X = rng.randn(60, 1)
        Y = np.sin(2 * X) + 0.05 * rng.randn(60, 1)
        m = DoublyStochasticDeepGP([1, 1, 1], Y, X, num_inducing=8, noise_var=0.05, seed=0)
        start = m.log_likelihood()
        hist = m.fit_minibatch(steps=60, batch_size=30, lr=0.02, seed=1, natgrad_gamma=0.2)
        m.set_batch(None)
        self.assertGreater(m.log_likelihood(), start)
        # the natural-gradient path must keep q(u) valid
        self.assertTrue(np.all(np.isfinite(m.layers[-1].q_chol)))
        m2 = DoublyStochasticDeepGP([1, 1, 1], Y, X, num_inducing=8, noise_var=0.05, seed=0)
        m2.fit_minibatch(steps=60, batch_size=30, lr=0.02, seed=1, natgrad_gamma=0.2)
        np.testing.assert_allclose(m2.optimizer_array, m.optimizer_array)


if __name__ == '__main__':
    unittest.main()
