# Copyright (c) 2015-2016, the authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""API and behaviour tests for DeepGP beyond the golden-file likelihood checks."""

import unittest

import numpy as np
import GPy

import deepgp


def _supervised_model(n=12, seed=0):
    rng = np.random.RandomState(seed)
    Y = rng.randn(n, 3)
    X = rng.randn(n, 2)
    kernels = [GPy.kern.RBF(4, ARD=True), GPy.kern.RBF(2, ARD=True)]
    return deepgp.DeepGP(
        [3, 4, 2],
        Y,
        X=X,
        kernels=kernels,
        num_inducing=3,
        back_constraint=False,
    ), X, Y


def _unsupervised_model(n=12, seed=1, dims=None, num_inducing=3):
    rng = np.random.RandomState(seed)
    if dims is None:
        dims = [3, 5, 2]
    Y = rng.randn(n, dims[0])
    kernels = [GPy.kern.RBF(d, ARD=True) for d in dims[1:]]
    return deepgp.DeepGP(
        dims,
        Y,
        kernels=kernels,
        num_inducing=num_inducing,
        back_constraint=False,
    ), Y


class DeepGPPredictionTest(unittest.TestCase):
    def setUp(self):
        self.model, self.X, self.Y = _supervised_model()

    def test_predict_shapes(self):
        mean, var = self.model.predict(self.X[:5])
        self.assertEqual(mean.shape, (5, 3))
        self.assertEqual(var.shape, (5, 3))
        self.assertTrue(np.all(np.isfinite(mean)))
        self.assertTrue(np.all(var > 0))

    def test_predict_with_samples_shapes(self):
        mean, var = self.model.predict_withSamples(self.X[:4], nSamples=8)
        self.assertEqual(mean.shape, (4, 3))
        self.assertEqual(var.shape, (4, 3))
        self.assertTrue(np.all(np.isfinite(mean)))

    def test_predict_quantiles_shapes(self):
        q_lo, q_hi = self.model.predict_quantiles(self.X[:3])
        self.assertEqual(q_lo.shape, (3, 3))
        self.assertEqual(q_hi.shape, (3, 3))
        self.assertTrue(np.all(q_lo <= q_hi))

    def test_y_x_properties(self):
        self.assertEqual(self.model.Y.shape, self.Y.shape)
        self.assertEqual(self.model.X.shape, self.X.shape)
        self.assertTrue(np.isfinite(float(self.model.log_likelihood())))


class DeepGPOptimizeTest(unittest.TestCase):
    def test_short_optimize_improves_or_holds(self):
        model, _, _ = _supervised_model(seed=2)
        before = float(model.log_likelihood())
        model.optimize(max_iters=3, messages=False)
        after = float(model.log_likelihood())
        self.assertTrue(np.isfinite(after))
        # A few L-BFGS steps should not make the bound substantially worse.
        self.assertGreaterEqual(after, before - 1e-3)


class DeepGPArchitectureTest(unittest.TestCase):
    def test_single_layer_checkgrad(self):
        model, _ = _unsupervised_model(dims=[3, 2], seed=3)
        self.assertEqual(model.nLayers, 1)
        self.assertTrue(model.checkgrad())

    def test_three_layer_checkgrad(self):
        model, _ = _unsupervised_model(dims=[3, 5, 3, 2], seed=4)
        self.assertEqual(model.nLayers, 3)
        self.assertTrue(model.checkgrad())

    def test_add_layer_from_single_observed(self):
        model, _ = _unsupervised_model(dims=[3, 2], seed=5, num_inducing=4)
        self.assertEqual(model.nLayers, 1)
        model.add_layer(2, back_constraint=False, num_inducing=4,
                        kernel=GPy.kern.RBF(2, ARD=True))
        self.assertEqual(model.nLayers, 2)
        self.assertTrue(model.checkgrad())

    def test_add_layer_from_existing_top(self):
        model, _, _ = _supervised_model(seed=6)
        self.assertEqual(model.nLayers, 2)
        model.add_layer(2, back_constraint=False, num_inducing=3,
                        kernel=GPy.kern.RBF(2, ARD=True))
        self.assertEqual(model.nLayers, 3)
        self.assertTrue(model.checkgrad())

    def test_invalid_obs_data_raises(self):
        Y = np.random.RandomState(0).randn(8, 3)
        with self.assertRaises(ValueError):
            deepgp.DeepGP(
                [3, 2],
                Y,
                kernels=[GPy.kern.RBF(2, ARD=True)],
                num_inducing=2,
                back_constraint=False,
                obs_data='nope',
            )

    def test_dimension_mismatch_raises(self):
        Y = np.random.RandomState(0).randn(8, 3)
        X = np.random.RandomState(0).randn(8, 2)
        with self.assertRaises(AssertionError):
            deepgp.DeepGP(
                [4, 2],
                Y,
                kernels=[GPy.kern.RBF(2, ARD=True)],
                num_inducing=2,
                back_constraint=False,
            )
        with self.assertRaises(AssertionError):
            deepgp.DeepGP(
                [3, 5, 3],
                Y,
                X=X,
                kernels=[GPy.kern.RBF(5, ARD=True), GPy.kern.RBF(3, ARD=True)],
                num_inducing=2,
                back_constraint=False,
            )


class DeepGPSVITest(unittest.TestCase):
    def test_svi_supervised_checkgrad(self):
        rng = np.random.RandomState(7)
        Y = rng.randn(12, 3)
        X = rng.randn(12, 2)
        model = deepgp.DeepGP(
            [3, 4, 2],
            Y,
            X=X,
            kernels=[GPy.kern.RBF(4, ARD=True), GPy.kern.RBF(2, ARD=True)],
            num_inducing=3,
            back_constraint=False,
            inference_method='svi',
        )
        self.assertTrue(model.checkgrad())
        self.assertTrue(np.isfinite(float(model.log_likelihood())))
        mean, var = model.predict(X[:3])
        self.assertEqual(mean.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(mean)))
        self.assertTrue(np.all(var > 0))


class MissingPredAPITest(unittest.TestCase):
    """PredLayer / PredDeepGP are not shipped; entry points should fail clearly."""

    def test_gen_pred_model_raises(self):
        model, _, _ = _supervised_model(seed=8)
        with self.assertRaises(NotImplementedError) as ctx:
            model.gen_pred_model()
        self.assertIn('issues/31', str(ctx.exception))

    def test_gen_pred_layer_raises(self):
        model, _, _ = _supervised_model(seed=9)
        with self.assertRaises(NotImplementedError) as ctx:
            model.layers[0].gen_pred_layer()
        self.assertIn('issues/31', str(ctx.exception))
