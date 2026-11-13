# Copyright (c) 2015-2016, the authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""CIP-0001 Phase 3: DeepGP psi_chunk_size / psi_threads wiring."""

import unittest

import numpy as np
import GPy

import deepgp


def _small_supervised(seed=0):
    rng = np.random.RandomState(seed)
    Y = rng.randn(24, 2)
    X = rng.randn(24, 1)
    kernels = [GPy.kern.RBF(3, ARD=True), GPy.kern.RBF(1, ARD=True)]
    return [2, 3, 1], Y, X, kernels


class PsiChunkingTest(unittest.TestCase):
    def test_chunked_matches_default_objective_and_grad(self):
        nDims, Y, X, kernels = _small_supervised()
        m0 = deepgp.DeepGP(
            nDims, Y, X=X, kernels=kernels, num_inducing=4, back_constraint=True
        )
        m1 = deepgp.DeepGP(
            nDims,
            Y,
            X=X,
            kernels=[k.copy() for k in kernels],
            num_inducing=4,
            back_constraint=True,
            psi_chunk_size=7,
            psi_threads=2,
        )
        m1.param_array[:] = m0.param_array
        m1.trigger_update()
        self.assertAlmostEqual(
            float(m0.log_likelihood()), float(m1.log_likelihood()), places=8
        )
        np.testing.assert_allclose(m0.gradient, m1.gradient, rtol=1e-8, atol=1e-8)
        for layer in m1.layers:
            self.assertEqual(layer.kern.psicomp.chunk_size, 7)
            self.assertEqual(layer.kern.psicomp.n_threads, 2)
            self.assertEqual(layer.inference_method.psi_chunk_size, 7)

    def test_checkgrad_with_chunking(self):
        nDims, Y, X, kernels = _small_supervised(seed=1)
        m = deepgp.DeepGP(
            nDims,
            Y,
            X=X,
            kernels=kernels,
            num_inducing=4,
            back_constraint=True,
            psi_chunk_size=5,
            psi_threads=2,
        )
        self.assertTrue(m.checkgrad())

    def test_defaults_leave_psicomp_unchunked(self):
        nDims, Y, X, kernels = _small_supervised(seed=2)
        m = deepgp.DeepGP(
            nDims, Y, X=X, kernels=kernels, num_inducing=4, back_constraint=True
        )
        for layer in m.layers:
            self.assertIsNone(layer.kern.psicomp.chunk_size)
            self.assertEqual(layer.kern.psicomp.n_threads, 1)


if __name__ == "__main__":
    unittest.main()
