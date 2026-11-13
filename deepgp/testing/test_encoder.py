# Copyright (c) 2015-2016, the authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""Unit tests for the NumPy MLP encoder (CIP-0001 Phase 1)."""

import unittest
import warnings

import numpy as np

from deepgp.encoder.mlp import MLP, MLP_model


class MLPModelGradientTest(unittest.TestCase):
    def _toy(self, activation='tanh', positive_obs=False, seed=0):
        rng = np.random.RandomState(seed)
        X = rng.randn(12, 3)
        Y = rng.randn(12, 2)
        if positive_obs:
            Y = np.abs(Y) + 0.1
        return MLP_model(
            X, Y, [3, 5, 2],
            activation=activation,
            positive_obs=positive_obs,
        )

    def test_checkgrad_tanh(self):
        self.assertTrue(self._toy('tanh', seed=1).checkgrad())

    def test_checkgrad_softplus(self):
        self.assertTrue(self._toy('softplus', seed=2).checkgrad())

    def test_checkgrad_positive_obs(self):
        self.assertTrue(self._toy('tanh', positive_obs=True, seed=3).checkgrad())

    def test_input_gradient_finite_difference(self):
        rng = np.random.RandomState(4)
        X = rng.randn(8, 3)
        dL = rng.randn(8, 2)
        mlp = MLP([3, 4, 2], activation='tanh')
        mlp.prepare()
        g_analytic = mlp.update_gradient(X, dL)

        eps = 1e-6
        g_num = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Xp, Xm = X.copy(), X.copy()
                Xp[i, j] += eps
                Xm[i, j] -= eps
                # Cost proxy: sum(pred * dL); gradient w.r.t. X matches update_gradient contract
                fp = np.sum(mlp.predict(Xp) * dL)
                fm = np.sum(mlp.predict(Xm) * dL)
                g_num[i, j] = (fp - fm) / (2 * eps)
        self.assertTrue(np.allclose(g_analytic, g_num, rtol=1e-5, atol=1e-6))

    def test_clone_round_trip(self):
        mlp = MLP([3, 4, 2], activation='tanh', X_center=np.array([0.1, -0.2, 0.0]))
        mlp.param_array[:] = np.linspace(0.1, 1.0, mlp.param_array.size)
        mlp._trigger_params_changed()
        clone = MLP.clone(mlp)
        X = np.random.RandomState(5).randn(5, 3)
        self.assertTrue(np.allclose(mlp.predict(X), clone.predict(X)))
        self.assertTrue(np.allclose(mlp.param_array, clone.param_array))

    def test_unsupported_activation_raises(self):
        with self.assertRaises(ValueError):
            MLP([3, 2], activation='relu')

    def test_regularization_warns_and_is_noop(self):
        X = np.random.RandomState(6).randn(6, 3)
        Y = np.random.RandomState(7).randn(6, 2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model = MLP_model(
                X, Y, [3, 4, 2],
                activation='tanh',
                regularization='L2',
                reg_weight=1.0,
            )
        self.assertTrue(any(issubclass(w.category, FutureWarning) for w in caught))
        self.assertTrue(model.checkgrad())
