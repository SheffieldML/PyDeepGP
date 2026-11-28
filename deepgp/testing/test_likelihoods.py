"""Non-Gaussian likelihoods: quadrature accuracy, derivatives, and bound gradients."""
import unittest

import numpy as np
from scipy.special import gammaln

from deepgp.functional import doubly_stochastic_bound
from deepgp.functional.likelihoods import Likelihood, Gaussian, Probit, Poisson, StudentT
from deepgp.testing.test_doubly_stochastic import _layer
from deepgp.testing.test_uncollapsed import _fd, _fd_kern


def _cases():
    rng = np.random.RandomState(0)
    mu, var = rng.randn(4, 2), 0.1 + rng.rand(4, 2)
    return [(Probit(), np.where(rng.rand(4, 2) > .5, 1., -1.), mu, var),
            (Poisson(), rng.poisson(2., (4, 2)).astype(float), 0.3 * mu, 0.3 * var),
            (StudentT(0.5, 3.), rng.randn(4, 2), mu, var)]


class QuadratureTest(unittest.TestCase):
    def test_generic_path_is_exact_for_gaussian(self):
        rng = np.random.RandomState(1)
        y, mu, var = rng.randn(5, 2), rng.randn(5, 2), 0.1 + rng.rand(5, 2)
        lik = Gaussian(0.3)
        got = Likelihood.variational_expectations(lik, y, mu, var)
        want = lik.variational_expectations(y, mu, var)
        for a, b in zip(got, want):
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)

    def test_poisson_matches_closed_form(self):
        rng = np.random.RandomState(2)
        y, mu, var = rng.poisson(3., (5, 2)).astype(float), 0.3 * rng.randn(5, 2), 0.2 * rng.rand(5, 2) + 0.05
        ell, _, _ = Poisson().variational_expectations(y, mu, var)
        np.testing.assert_allclose(ell, -np.exp(mu + var / 2) + y * mu - gammaln(y + 1), rtol=1e-8)

    def test_derivatives_match_finite_differences(self):
        h = 1e-6
        for lik, y, mu, var in _cases():
            ell, gm, gv = lik.variational_expectations(y, mu, var)
            ep, _, _ = lik.variational_expectations(y, mu + h, var)
            em, _, _ = lik.variational_expectations(y, mu - h, var)
            np.testing.assert_allclose(gm, (ep - em) / (2 * h), rtol=1e-5, atol=1e-7, err_msg=type(lik).__name__)
            ep, _, _ = lik.variational_expectations(y, mu, var + h)
            em, _, _ = lik.variational_expectations(y, mu, var - h)
            np.testing.assert_allclose(gv, (ep - em) / (2 * h), rtol=1e-5, atol=1e-7, err_msg=type(lik).__name__)

    def test_zero_variance_is_the_log_density(self):
        for lik, y, mu, var in _cases():
            ell, gm, gv = lik.variational_expectations(y, mu, np.zeros_like(var))
            np.testing.assert_allclose(ell, lik.log_density(y, mu), rtol=1e-10)
            np.testing.assert_allclose(gm, lik.d_log_density(y, mu), rtol=1e-10)
            np.testing.assert_array_equal(gv, 0.)

    def test_probit_predictive_mean_is_the_quadrature_probability(self):
        rng = np.random.RandomState(3)
        mu, var = rng.randn(6), 0.1 + rng.rand(6)
        lik = Probit()
        np.testing.assert_allclose(lik.predictive_mean(mu, var), lik.predictive_density(np.ones(6), mu, var),
                                   rtol=1e-8)
        np.testing.assert_allclose(lik.predictive_density(np.ones(6), mu, var)
                                   + lik.predictive_density(-np.ones(6), mu, var), 1., rtol=1e-8)

    def test_validation(self):
        with self.assertRaises(ValueError):
            Probit().check(np.array([0., 1.]))
        with self.assertRaises(ValueError):
            Poisson().check(np.array([-1., 2.]))
        with self.assertRaises(ValueError):
            Poisson().check(np.array([0.5]))
        with self.assertRaises(ValueError):
            StudentT(-1., 3.)


class BoundTest(unittest.TestCase):
    def _stack(self):
        rng = np.random.RandomState(4)
        N = 6
        X = rng.randn(N, 2)
        layers = [_layer(rng, 2, 2, 4, residual=True), _layer(rng, 2, 1, 4)]
        eps = [[rng.randn(N, 2)] for _ in range(2)]
        return layers, X, eps, rng

    def test_gaussian_object_equals_analytic_bound(self):
        layers, X, eps, rng = self._stack()
        Y = rng.randn(6, 1)
        a, ga = doubly_stochastic_bound(layers, X, Y, 0.4, eps, scale=2.)
        b, gb = doubly_stochastic_bound(layers, X, Y, 0.4, eps, scale=2., likelihood=Gaussian(0.4))
        self.assertAlmostEqual(a, b, places=8)
        np.testing.assert_allclose(ga['layers'][1]['q_mu'], gb['layers'][1]['q_mu'], rtol=1e-8)
        self.assertEqual(gb['noise_var'], 0.)

    def test_gradients_for_each_likelihood(self):
        layers, X, eps, rng = self._stack()
        data = {'probit': (Probit(), np.where(rng.rand(6, 1) > .5, 1., -1.)),
                'poisson': (Poisson(), rng.poisson(2., (6, 1)).astype(float)),
                'student': (StudentT(0.3, 4.), rng.randn(6, 1))}
        tol = dict(rtol=1e-5, atol=1e-6)
        for name, (lik, Y) in data.items():
            F = lambda: doubly_stochastic_bound(layers, X.copy(), Y, 1., eps, scale=2., likelihood=lik)[0]
            _, g = doubly_stochastic_bound(layers, X.copy(), Y, 1., eps, scale=2., likelihood=lik)
            for l, layer in enumerate(layers):
                np.testing.assert_allclose(g['layers'][l]['q_mu'], _fd(F, layer['q_mu']), err_msg=name, **tol)
                np.testing.assert_allclose(g['layers'][l]['Z'], _fd(F, layer['Z']), err_msg=name, **tol)
                np.testing.assert_allclose(g['layers'][l]['kern'], _fd_kern(F, layer['kern']),
                                           err_msg=name, **tol)
            np.testing.assert_allclose(g['X'], _fd(F, X), err_msg=name, **tol)


if __name__ == '__main__':
    unittest.main()
