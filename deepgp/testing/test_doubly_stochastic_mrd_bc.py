"""Multiple views (MRD) and back-constrained q(X) for the doubly stochastic model."""
import unittest

import numpy as np

from deepgp.functional import doubly_stochastic_bound
from deepgp.models.uncollapsed import DoublyStochasticDeepGP


def _views(N=36, seed=0):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 6, N)[:, None]
    A = np.hstack([np.sin(t), np.cos(t)]) + 0.05 * rng.randn(N, 2)
    B = np.hstack([np.sin(2 * t), t / 3.]) + 0.05 * rng.randn(N, 2)
    C = np.sin(t * 1.5) + 0.05 * rng.randn(N, 1)
    return A, B, C


def _perturb(m, seed=1):
    rng = np.random.RandomState(seed)
    for h in m.heads:
        h.q_mu[:] = 0.3 * rng.randn(*h.q_mu.shape)


class FunctionalTest(unittest.TestCase):
    def test_single_view_matches_stack_signature(self):
        rng = np.random.RandomState(0)
        m = DoublyStochasticDeepGP([2, 2, 1], _views()[0], None, num_inducing=6, seed=0)
        layers = [l.as_dict() for l in m.layers]
        X, x_var = m.X_mean.values, m.X_var.values
        a, ga = doubly_stochastic_bound(layers, X, m.Y_all, 0.1, m.eps, x_var=x_var, eps_x=m.eps_x)
        views = [dict(layer=layers[-1], Y=m.Y_all, noise_var=0.1)]
        b, gb = doubly_stochastic_bound(layers[:-1], X, None, None, m.eps, x_var=x_var,
                                        eps_x=m.eps_x, views=views)
        self.assertAlmostEqual(a, b, places=10)
        np.testing.assert_allclose(ga['X'], gb['X'])


class MRDTest(unittest.TestCase):
    def _model(self, nDims=None, **kw):
        A, B, _ = _views()
        return DoublyStochasticDeepGP(nDims or [[2, 2], 1], [A, B], None, num_inducing=6,
                                      noise_var=0.05, seed=0, **kw)

    def test_structure(self):
        m = self._model([[2, 2], 2, 1])
        self.assertEqual(len(m.heads), 2)
        self.assertEqual(m.n_trunk, 1)
        self.assertEqual(m.noise_variance.shape, (2,))
        self.assertEqual(m.X_mean.shape, (36, 1))

    def test_checkgrad_heads_on_latent(self):
        m = self._model()
        _perturb(m)
        self.assertTrue(m.checkgrad())

    def test_checkgrad_shared_trunk_and_minibatch(self):
        m = self._model([[2, 2], 2, 1])
        _perturb(m)
        m.set_batch(np.arange(0, 36, 3))
        self.assertTrue(m.checkgrad())

    def test_training_and_per_view_prediction(self):
        A, B, _ = _views()
        m = self._model()
        m.set_batch(None)
        start = m.log_likelihood()
        m.fit_minibatch(steps=150, batch_size=18, lr=0.02, seed=3, natgrad_gamma=0.05)
        m.set_batch(None)
        self.assertGreater(m.log_likelihood(), start)
        X = m.X_mean.values
        for v, Yv in enumerate((A, B)):
            mean, var = m.predict(X, view=v)
            self.assertEqual(mean.shape, Yv.shape)
            self.assertLess(np.sqrt(np.mean((mean - Yv) ** 2)), np.sqrt(np.mean((Yv - Yv.mean(0)) ** 2)))
            self.assertEqual(m.log_predictive_density(X, Yv, view=v).shape, (36,))

    def test_validation(self):
        A, B, _ = _views()
        with self.assertRaises(ValueError):
            DoublyStochasticDeepGP([[2, 2], 1], [A, B[:30]], None)
        with self.assertRaises(ValueError):
            DoublyStochasticDeepGP([[2, 3], 1], [A, B], None)
        with self.assertRaises(ValueError):
            DoublyStochasticDeepGP([2, 1], [A, B], None)
        from deepgp.functional.likelihoods import Poisson
        with self.assertRaises(ValueError):
            DoublyStochasticDeepGP([[2, 2], 1], [A, B], None, likelihood=Poisson())


class BackConstraintTest(unittest.TestCase):
    def test_encoder_replaces_free_means(self):
        A, _, _ = _views()
        m = DoublyStochasticDeepGP([2, 1], A, None, num_inducing=6, seed=0, back_constraint=True)
        self.assertFalse(hasattr(m, 'X_mean'))
        self.assertEqual(len(m.encoders), 1)
        self.assertEqual(m.latent_mean().shape, (36, 1))
        self.assertEqual(m.encode(A[:5]).shape, (5, 1))

    def test_encoder_starts_near_pca(self):
        A, _, _ = _views()
        m = DoublyStochasticDeepGP([2, 1], A, None, num_inducing=6, seed=0, back_constraint=True)
        corr = abs(np.corrcoef(m.latent_mean()[:, 0], m.X_all[:, 0])[0, 1])
        self.assertGreater(corr, 0.9)

    def test_checkgrad_and_minibatch(self):
        A, _, _ = _views()
        m = DoublyStochasticDeepGP([2, 1], A, None, num_inducing=6, seed=0, back_constraint=True)
        _perturb(m)
        self.assertTrue(m.checkgrad())
        m.set_batch(np.arange(0, 36, 3))
        self.assertTrue(m.checkgrad())

    def test_checkgrad_mrd_product_of_experts(self):
        A, B, _ = _views()
        m = DoublyStochasticDeepGP([[2, 2], 2, 1], [A, B], None, num_inducing=6, seed=0,
                                   back_constraint=True)
        self.assertEqual(len(m.encoders), 2)
        _perturb(m)
        self.assertTrue(m.checkgrad())
        m.set_batch(np.arange(1, 36, 4))
        self.assertTrue(m.checkgrad())

    def test_training_improves_bound(self):
        A, B, _ = _views()
        m = DoublyStochasticDeepGP([[2, 2], 1], [A, B], None, num_inducing=6, seed=0,
                                   back_constraint=True)
        m.set_batch(None)
        start = m.log_likelihood()
        m.fit_minibatch(steps=100, batch_size=18, lr=0.02, seed=3)
        m.set_batch(None)
        self.assertGreater(m.log_likelihood(), start)

    def test_validation(self):
        A, _, _ = _views()
        with self.assertRaises(ValueError):
            DoublyStochasticDeepGP([2, 1], A, np.zeros((36, 1)), back_constraint=True)
        m = DoublyStochasticDeepGP([2, 1], A, None, num_inducing=4)
        with self.assertRaises(ValueError):
            m.encode(A)


if __name__ == '__main__':
    unittest.main()
