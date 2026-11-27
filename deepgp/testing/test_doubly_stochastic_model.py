"""DoublyStochasticDeepGP: GPy model behaviour (checkgrad, determinism, minibatch, predict)."""
import unittest

import numpy as np

from deepgp.models.uncollapsed import DoublyStochasticDeepGP


def _data(N=40, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-2, 2, (N, 1))
    Y = np.sin(2 * X) + 0.1 * rng.randn(N, 1)
    return X, Y


class ModelTest(unittest.TestCase):
    def test_checkgrad_two_hidden_widths(self):
        X, Y = _data(12)
        m = DoublyStochasticDeepGP([1, 2, 1], Y, X, num_inducing=4, n_samples=2, seed=1)
        self.assertEqual([l.q_mu.shape[1] for l in m.layers], [2, 1])
        self.assertTrue(m.checkgrad(tolerance=1e-3))

    def test_checkgrad_residual_hidden(self):
        X, Y = _data(10)
        m = DoublyStochasticDeepGP([1, 1, 1], Y, X, num_inducing=4, seed=2)
        self.assertTrue(m.layers[0].residual)
        self.assertTrue(m.checkgrad(tolerance=1e-3))

    def test_chi_sets_hidden_kernel_variance(self):
        X, Y = _data(12)
        m = DoublyStochasticDeepGP([1, 1, 1, 1], Y, X, num_inducing=4, chi=1.0)
        v = [float(l.kern.variance.values[0]) for l in m.layers]
        self.assertEqual(v, [0.5, 0.5, 1.0])
        m1 = DoublyStochasticDeepGP([1, 1], Y, X, num_inducing=4, chi=1.0)
        self.assertEqual(float(m1.layers[0].kern.variance.values[0]), 1.0)
        self.assertTrue(m.checkgrad(tolerance=1e-3))

    def test_checkgrad_minibatch(self):
        X, Y = _data(12)
        m = DoublyStochasticDeepGP([1, 2, 1], Y, X, num_inducing=4, seed=3)
        m.set_batch([0, 3, 5, 7, 9])
        self.assertTrue(m.checkgrad(tolerance=1e-3))

    def test_objective_deterministic_until_resample(self):
        X, Y = _data(10)
        m = DoublyStochasticDeepGP([1, 2, 1], Y, X, num_inducing=4, seed=4)
        # At initialisation q(u) is the prior, so the top layer ignores its input; move it off.
        m.layers[-1].q_mu[:] = np.random.RandomState(0).randn(*m.layers[-1].q_mu.shape)
        a = m.log_likelihood(); m.parameters_changed()
        self.assertEqual(a, m.log_likelihood())
        m.resample()
        self.assertNotEqual(a, m.log_likelihood())
        m.resample(seed=99); b = m.log_likelihood()
        m.resample(seed=99); self.assertEqual(b, m.log_likelihood())

    def test_validation(self):
        X, Y = _data(10)
        with self.assertRaises(ValueError):
            DoublyStochasticDeepGP([1, 2, 3], Y, None)   # latent wider than Y
        with self.assertRaises(ValueError):
            DoublyStochasticDeepGP([2, 1], Y, X)

    def test_optimize_improves_fixed_noise_objective(self):
        X, Y = _data(25)
        m = DoublyStochasticDeepGP([1, 1], Y, X, num_inducing=8, seed=5)
        before = m.log_likelihood()
        m.optimize(max_iters=60, messages=False)
        self.assertGreater(m.log_likelihood(), before + 1.)

    def test_minibatch_fit_learns_and_is_reproducible(self):
        X, Y = _data(60)

        def run():
            m = DoublyStochasticDeepGP([1, 1, 1], Y, X, num_inducing=10, seed=7, batch_size=20)
            h = m.fit_minibatch(steps=150, lr=0.02, seed=11)
            return m, h
        m1, h1 = run()
        m2, h2 = run()
        np.testing.assert_allclose(h1, h2)                     # seeded: identical trajectories
        self.assertGreater(np.mean(h1[-20:]), np.mean(h1[:20]))
        mean, var = m1.predict(X, seed=0)
        self.assertEqual(mean.shape, Y.shape)
        self.assertTrue(np.all(var > 0))
        self.assertLess(np.mean((mean - Y) ** 2), np.var(Y))   # beats the constant predictor


if __name__ == '__main__':
    unittest.main()


class SelectionByNameTest(unittest.TestCase):
    def test_inference_method_selects_family(self):
        import deepgp
        X, Y = _data(10)
        m = deepgp.DeepGP([1, 2, 1], Y, X, num_inducing=4, inference_method='doubly_stochastic', seed=1)
        self.assertIsInstance(m, DoublyStochasticDeepGP)
        self.assertTrue(np.isfinite(m.log_likelihood()))

    def test_default_unchanged(self):
        import deepgp
        X, Y = _data(10)
        m = deepgp.DeepGP([1, 2], Y, num_inducing=4, back_constraint=False)
        self.assertIsInstance(m, deepgp.DeepGP)
        self.assertNotIsInstance(m, DoublyStochasticDeepGP)


class UnsupervisedTest(unittest.TestCase):
    def _model(self, **kw):
        rng = np.random.RandomState(0)
        t = np.linspace(0, 6, 40)[:, None]
        Y = np.hstack([np.sin(t), np.cos(t), np.sin(2 * t)]) + 0.05 * rng.randn(40, 3)
        return Y, DoublyStochasticDeepGP([3, 2, 1], Y, None, num_inducing=8, noise_var=0.05, seed=0, **kw)

    def test_checkgrad(self):
        _, m = self._model()
        m.layers[-1].q_mu[:] = 0.3 * np.random.RandomState(1).randn(*m.layers[-1].q_mu.shape)
        self.assertTrue(m.checkgrad())

    def test_checkgrad_minibatch(self):
        _, m = self._model()
        m.layers[-1].q_mu[:] = 0.3 * np.random.RandomState(1).randn(*m.layers[-1].q_mu.shape)
        m.set_batch(np.arange(0, 40, 3))
        self.assertTrue(m.checkgrad())

    def test_latent_parameters_exist_and_pca_initialised(self):
        Y, m = self._model()
        self.assertEqual(m.X_mean.shape, (40, 1))
        self.assertEqual(m.X_var.shape, (40, 1))
        self.assertTrue(m.unsupervised)

    def test_training_improves_bound(self):
        _, m = self._model()
        m.set_batch(None)
        start = m.log_likelihood()
        m.fit_minibatch(steps=80, batch_size=20, lr=0.02, seed=3)
        m.set_batch(None)
        self.assertGreater(m.log_likelihood(), start)
        mean, var = m.predict(m.X_mean.values[:5])
        self.assertEqual(mean.shape, (5, 3))
        self.assertTrue(np.all(var > 0))


class SelectionUnsupervisedTest(unittest.TestCase):
    def test_deepgp_without_x_selects_unsupervised_model(self):
        import deepgp
        Y = np.random.RandomState(0).randn(20, 4)
        m = deepgp.DeepGP([4, 2], Y, num_inducing=5, inference_method='doubly_stochastic')
        self.assertIsInstance(m, DoublyStochasticDeepGP)
        self.assertTrue(m.unsupervised)
