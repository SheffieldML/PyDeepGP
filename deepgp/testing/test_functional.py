# Copyright (c) 2015-2016, the authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""CIP-0003 Phase 2: pure layer functions and layer checkgrad."""

import unittest

import numpy as np
import GPy

import deepgp
from deepgp.functional import collapsed_bound, route_gradients


def _stats(seed=0, N=12, D=2, M=4, Q=2, uncertain_inputs=True):
    rng = np.random.RandomState(seed)
    Z = rng.randn(M, Q)
    k = GPy.kern.RBF(Q, ARD=True)
    Kmm = k.K(Z) + 1e-6 * np.eye(M)
    psi1 = rng.rand(N, M)
    psi2 = psi1.T @ psi1 + 0.1 * np.eye(M) if uncertain_inputs else None
    psi0 = np.ones(N) * 1.2
    return Kmm, psi0, psi1, psi2, rng.randn(N, D), rng.rand(N, D) * .1 + .02


def _fd(f, x, eps=1e-6):
    g = np.zeros_like(x)
    for i in np.ndindex(*x.shape):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2 * eps)
    return g


class CollapsedBoundTest(unittest.TestCase):
    def test_gradients_wrt_statistics_and_outputs(self):
        Kmm, psi0, psi1, psi2, Ym, Yv = _stats()
        beta = 5.
        r = collapsed_bound(Kmm, psi0, psi1, psi2, Ym, beta, Y_var=Yv)
        f = lambda **kw: float(collapsed_bound(**{**dict(
            Kmm=Kmm, psi0=psi0, psi1=psi1, psi2=psi2, Y_mean=Ym, beta=beta, Y_var=Yv), **kw})['logL'][0])
        np.testing.assert_allclose(r['dL_dpsi1'], _fd(lambda x: f(psi1=x), psi1), atol=1e-5)
        np.testing.assert_allclose(r['dL_dYmean'], _fd(lambda x: f(Y_mean=x), Ym), atol=1e-5)
        # The bound sees only row sums of Y_var, so dL_dYvar is per row.
        g = _fd(lambda x: f(Y_var=x), Yv)
        np.testing.assert_allclose(g, np.repeat(r['dL_dYvar'][:, None], Yv.shape[1], 1), atol=1e-5)
        # dL_dpsi0 is per point: shift one entry.
        g0 = _fd(lambda x: f(psi0=x), psi0)
        np.testing.assert_allclose(r['dL_dpsi0'], g0, atol=1e-5)

    def test_deterministic_inputs_fold_psi2_into_psi1(self):
        Kmm, psi0, psi1, _, Ym, _ = _stats(uncertain_inputs=False)
        r = collapsed_bound(Kmm, psi0, psi1, None, Ym, 4.)
        self.assertIsNone(r['dL_dpsi2'])
        g = _fd(lambda x: float(collapsed_bound(Kmm, psi0, x, None, Ym, 4.)['logL'][0]), psi1)
        np.testing.assert_allclose(r['dL_dpsi1'], g, atol=1e-5)

    def test_matches_gpy_for_certain_outputs(self):
        from GPy.inference.latent_function_inference.var_dtc import VarDTC as G
        rng = np.random.RandomState(4)
        Z, X, Y = rng.randn(5, 2), rng.randn(20, 2), rng.randn(20, 3)
        k = GPy.kern.RBF(2, ARD=True)
        lik = GPy.likelihoods.Gaussian(variance=.2)
        g = G(); g.const_jitter = 1e-6
        _, lg, _ = g.inference(k, X, Z, lik, Y)
        Kmm = k.K(Z) + 1e-6 * np.eye(5)
        r = collapsed_bound(Kmm, k.Kdiag(X), k.K(X, Z), None, Y, 1. / .2)
        np.testing.assert_allclose(r['logL'][0], np.squeeze(lg), rtol=1e-10)

    def test_route_gradients_returns_arrays(self):
        rng = np.random.RandomState(5)
        Z, X = rng.randn(4, 2), rng.randn(9, 2)
        k = GPy.kern.RBF(2, ARD=True)
        kg, zg = route_gradients(k, Z, X, np.eye(4), dL_dKdiag=np.ones(9),
                                 dL_dKnm=rng.randn(9, 4))
        self.assertEqual(kg.shape, (k.size,))
        self.assertEqual(zg.shape, Z.shape)
        self.assertIsInstance(kg, np.ndarray)
        self.assertNotIsInstance(zg, GPy.core.Param)


class LayerCheckgradTest(unittest.TestCase):
    def _data(self):
        rng = np.random.RandomState(7)
        return rng.randn(10, 3), rng.randn(10, 1)

    def test_unsupervised_stack(self):
        Y, _ = self._data()
        m = deepgp.DeepGP([3, 2, 2], Y, kernels=[GPy.kern.RBF(2, ARD=True), GPy.kern.RBF(2, ARD=True)],
                          num_inducing=4, back_constraint=False)
        self.assertTrue(m.checkgrad())

    def test_supervised_stack(self):
        Y, X = self._data()
        m = deepgp.DeepGP([3, 2, 1], Y, X=X, kernels=[GPy.kern.RBF(2, ARD=True), GPy.kern.RBF(1, ARD=True)],
                          num_inducing=4, back_constraint=False)
        self.assertTrue(m.checkgrad())

    def test_back_constrained_stack(self):
        Y, _ = self._data()
        m = deepgp.DeepGP([3, 2, 2], Y, kernels=[GPy.kern.RBF(2, ARD=True), GPy.kern.RBF(2, ARD=True)],
                          num_inducing=4, back_constraint=True)
        self.assertTrue(m.checkgrad())


if __name__ == '__main__':
    unittest.main()
