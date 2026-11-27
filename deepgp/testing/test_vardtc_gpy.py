# Copyright (c) 2015-2016, the authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""CIP-0003 Phase 2: deepgp VarDTC is GPy's VarDTC plus a documented delta."""

import unittest

import numpy as np
import GPy
from GPy.core.parameterization.variational import NormalPosterior
from GPy.inference.latent_function_inference.var_dtc import VarDTC as GPyVarDTC

from deepgp.inference import VarDTC


def _problem(uncertain, seed=0, N=30, D=3, Q=2, M=6):
    rng = np.random.RandomState(seed)
    Y = rng.randn(N, D)
    Z = rng.randn(M, Q)
    X = (NormalPosterior(rng.randn(N, Q), rng.rand(N, Q) * .1 + .05)
         if uncertain else rng.randn(N, Q))
    return Y, X, Z


class VarDTCDeltaTest(unittest.TestCase):
    def setUp(self):
        self.lik = GPy.likelihoods.Gaussian(variance=.1)
        self.kern = GPy.kern.RBF(2, ARD=True)

    def _gpy(self):
        g = GPyVarDTC()
        g.const_jitter = 1e-6
        return g

    def test_agrees_with_gpy_for_deterministic_outputs(self):
        for uncertain in (False, True):
            Y, X, Z = _problem(uncertain)
            pg, lg, gg = self._gpy().inference(self.kern, X, Z, self.lik, Y)
            pl, ll, gl = VarDTC().inference(self.kern, X, Z, self.lik, Y)
            np.testing.assert_allclose(np.squeeze(ll), np.squeeze(lg), rtol=1e-10)
            for k, v in gl.items():
                np.testing.assert_allclose(v, np.reshape(gg[k], np.shape(v)), atol=1e-8)
            np.testing.assert_allclose(pl.woodbury_vector, pg.woodbury_vector, atol=1e-9)

    def test_chunked_psi_matches_unchunked(self):
        Y, X, Z = _problem(True)
        _, l0, g0 = VarDTC().inference(self.kern, X, Z, self.lik, Y)
        _, l1, g1 = VarDTC(psi_chunk_size=7).inference(self.kern, X, Z, self.lik, Y)
        np.testing.assert_allclose(l1, l0, rtol=1e-10)
        for k in g0:
            np.testing.assert_allclose(g1[k], g0[k], atol=1e-8)

    def test_kuu_sigma_changes_bound_consistently(self):
        Y, X, Z = _problem(True)
        sig = np.full(Z.shape[0], 1e-3)
        _, l0, _ = VarDTC().inference(self.kern, X, Z, self.lik, Y)
        _, l1, _ = VarDTC().inference(self.kern, X, Z, self.lik, Y, Kuu_sigma=sig)
        self.assertTrue(np.isfinite(l1).all())
        self.assertGreater(abs(float(l1[0] - l0[0])), 0.0)

    def test_uncertain_outputs_supported(self):
        Y, X, Z = _problem(True)
        Yq = NormalPosterior(Y, np.full(Y.shape, .05))
        _, ll, gl = VarDTC().inference(self.kern, X, Z, self.lik, Yq)
        self.assertTrue(np.isfinite(ll).all())
        self.assertIn('dL_dYmean', gl)
        # With vanishing output variance, the bound approaches the deterministic one.
        Yq0 = NormalPosterior(Y, np.full(Y.shape, 1e-12))
        _, l0, _ = VarDTC().inference(self.kern, X, Z, self.lik, Yq0)
        _, lD, _ = VarDTC().inference(self.kern, X, Z, self.lik, Y)
        np.testing.assert_allclose(l0, lD, rtol=1e-6)

    def test_sparse_linalg_modes_agree(self):
        Y, X, Z = _problem(False)
        ref = VarDTC().inference(self.kern, X, Z, self.lik, Y)[1]
        try:
            lq = VarDTC(sparse_linalg='lq').inference(self.kern, X, Z, self.lik, Y)[1]
        except TypeError:
            self.skipTest('GPy without CIP-0013 sparse_linalg')
        np.testing.assert_allclose(lq, ref, rtol=1e-6)

    def test_layer_exposes_sparse_linalg(self):
        import deepgp
        Y = np.random.RandomState(2).randn(10, 2)
        try:
            layer = deepgp.layers.ObservedLayer(2, 1, Y, num_inducing=3,
                                                back_constraint=False)
            layer.inference_method.sparse_linalg = 'lq'
        except (TypeError, AttributeError):
            self.skipTest('GPy without CIP-0013 sparse_linalg')
        self.assertEqual(layer.inference_method.sparse_linalg, 'lq')

    def test_inferentia_fails_clearly(self):
        import deepgp
        Y = np.random.RandomState(3).randn(10, 2)
        with self.assertRaises(NotImplementedError):
            deepgp.layers.ObservedLayer(2, 1, Y, num_inducing=3, back_constraint=False,
                                        inference_method='inferentia')


if __name__ == '__main__':
    unittest.main()
