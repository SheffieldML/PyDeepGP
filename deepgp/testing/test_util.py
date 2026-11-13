# Copyright (c) 2015-2016, the authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""Unit tests for deepgp.util helpers used by DeepGP construction and SVI."""

import unittest

import numpy as np
import GPy
from GPy.core.parameterization.variational import NormalPosterior

import deepgp
from deepgp.util.misc import comp_mapping, gen_timestring
from deepgp.util.util import check_snr, initialize_latent, linsp, transform_labels
from deepgp.util.variational import BernoulliEntropy, NormalEntropy, NormalPrior


class InitializeLatentTest(unittest.TestCase):
    def setUp(self):
        self.Y = np.random.RandomState(0).randn(15, 4)

    def test_pca(self):
        X, var = initialize_latent('PCA', self.Y.shape[0], 2, self.Y)
        self.assertEqual(X.shape, (15, 2))
        self.assertEqual(var.shape, (2,))
        self.assertTrue(np.allclose(X.mean(0), 0.0, atol=1e-10))

    def test_random_projection(self):
        X, var = initialize_latent('randomProjection', self.Y.shape[0], 3, self.Y)
        self.assertEqual(X.shape, (15, 3))
        self.assertEqual(var.shape, (3,))
        self.assertTrue(np.all(np.isfinite(X)))

    def test_random_default(self):
        X, var = initialize_latent('rand', self.Y.shape[0], 2, self.Y)
        self.assertEqual(X.shape, (15, 2))
        self.assertEqual(var.shape, (2,))


class LabelAndMiscUtilTest(unittest.TestCase):
    def test_transform_labels_onehot(self):
        labels = np.array([[0], [2], [1], [2]])
        onehot = transform_labels(labels)
        self.assertEqual(onehot.shape, (4, 3))
        self.assertEqual(onehot.sum(), 4)
        self.assertTrue(np.array_equal(onehot.argmax(1), labels.ravel()))

    def test_transform_labels_argmax(self):
        onehot = np.eye(3)
        labels = transform_labels(onehot)
        self.assertEqual(labels.shape, (3, 1))
        self.assertTrue(np.array_equal(labels.ravel(), [0, 1, 2]))

    def test_linsp(self):
        self.assertTrue(np.allclose(linsp(1, 4), [1, 2, 3, 4]))

    def test_gen_timestring(self):
        plain = gen_timestring()
        named = gen_timestring('proj')
        self.assertRegex(plain, r'^\d{4}:\d{2}:\d{2}_')
        self.assertTrue(named.startswith('proj_'))

    def test_comp_mapping(self):
        rng = np.random.RandomState(1)
        X = rng.randn(10, 2)
        W_true = rng.randn(2, 3)
        Y = X.dot(W_true)
        W = comp_mapping(X, Y)
        self.assertEqual(W.shape, (2, 3))
        self.assertTrue(np.allclose(W, W_true, atol=1e-8))

    def test_comp_mapping_variational(self):
        rng = np.random.RandomState(2)
        X = NormalPosterior(rng.randn(8, 2), np.ones((8, 2)) * 0.01)
        Y = NormalPosterior(rng.randn(8, 3), np.ones((8, 3)) * 0.01)
        W = comp_mapping(X, Y)
        self.assertEqual(W.shape, (2, 3))


class VariationalHelpersTest(unittest.TestCase):
    def setUp(self):
        mean = np.random.RandomState(3).randn(6, 2)
        var = np.ones((6, 2)) * 0.2
        self.q = NormalPosterior(mean, var)

    def test_normal_entropy(self):
        term = NormalEntropy()
        value = term.comp_value(self.q)
        self.assertTrue(np.isfinite(value))
        self.q.variance.gradient[:] = 0.0
        term.update_gradients(self.q)
        self.assertTrue(np.all(self.q.variance.gradient > 0))

    def test_normal_prior(self):
        term = NormalPrior()
        value = term.comp_value(self.q)
        self.assertTrue(np.isfinite(value))
        self.q.mean.gradient[:] = 0.0
        self.q.variance.gradient[:] = 0.0
        term.update_gradients(self.q)
        self.assertTrue(np.all(np.isfinite(self.q.mean.gradient)))
        self.assertTrue(np.all(np.isfinite(self.q.variance.gradient)))

    def test_bernoulli_entropy(self):
        term = BernoulliEntropy()
        qS = np.clip(np.random.RandomState(4).rand(5, 1), 0.05, 0.95)
        value = term.comp_value(qS)
        self.assertTrue(np.isfinite(value))


class CheckSNRTest(unittest.TestCase):
    def test_check_snr_on_supervised_model(self):
        rng = np.random.RandomState(5)
        Y = rng.randn(10, 3)
        X = rng.randn(10, 2)
        model = deepgp.DeepGP(
            [3, 4, 2],
            Y,
            X=X,
            kernels=[GPy.kern.RBF(4, ARD=True), GPy.kern.RBF(2, ARD=True)],
            num_inducing=3,
            back_constraint=False,
        )
        snr = check_snr(model, messages=False)
        self.assertIsInstance(snr, list)
        self.assertGreaterEqual(len(snr), 1)
        self.assertTrue(np.all(np.asarray(snr[0]) > 0))
