# Copyright (c) 2015-2016, the authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""CIP-0002: multi-view / MRD DeepGP construction and optimisation tests."""

import unittest

import numpy as np
import GPy

import deepgp
from deepgp.layers.mrd import ObservedMRDLayer


def _correlated_views(n=24, q=3, d=6, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, q)
    Y1 = X @ rng.randn(q, d) + 0.05 * rng.randn(n, d)
    Y2 = X @ rng.randn(q, d) + 0.05 * rng.randn(n, d)
    return Y1, Y2


def _mrd_kernels(q_shared, q_top=None):
    k_obs = [
        GPy.kern.RBF(q_shared, ARD=True) + GPy.kern.Bias(q_shared),
        GPy.kern.RBF(q_shared, ARD=True) + GPy.kern.Bias(q_shared),
    ]
    if q_top is None:
        return k_obs
    return [k_obs, GPy.kern.RBF(q_top, ARD=True) + GPy.kern.Bias(q_top)]


class MRDConstructTest(unittest.TestCase):
    def test_two_view_constructs_finite_bound(self):
        Y1, Y2 = _correlated_views()
        q1, q2 = 4, 3
        m = deepgp.DeepGP(
            [[Y1.shape[1], Y2.shape[1]], q1, q2],
            [Y1, Y2],
            kernels=_mrd_kernels(q1, q2),
            num_inducing=8,
            back_constraint=False,
        )
        ll = float(m.log_likelihood())
        self.assertTrue(np.isfinite(ll))
        self.assertEqual(np.asarray(m.layers[0]._log_marginal_likelihood).shape, (1,))

    def test_optimize_smoke(self):
        Y1, Y2 = _correlated_views(seed=1)
        q1, q2 = 4, 3
        m = deepgp.DeepGP(
            [[Y1.shape[1], Y2.shape[1]], q1, q2],
            [Y1, Y2],
            kernels=_mrd_kernels(q1, q2),
            num_inducing=8,
            back_constraint=False,
        )
        before = float(m.log_likelihood())
        m.optimize(max_iters=5, messages=False)
        after = float(m.log_likelihood())
        self.assertTrue(np.isfinite(after))
        self.assertGreaterEqual(after, before - 1.0)

    def test_kernel_list_required(self):
        Y1, Y2 = _correlated_views(n=12, d=4, seed=2)
        q1 = 3
        with self.assertRaises(AssertionError):
            deepgp.DeepGP(
                [[Y1.shape[1], Y2.shape[1]], q1],
                [Y1, Y2],
                kernels=[GPy.kern.RBF(q1, ARD=True)],  # not a per-view list
                num_inducing=5,
                back_constraint=False,
            )


class MRDInitTest(unittest.TestCase):
    def test_pca_concat_and_single(self):
        Y1, Y2 = _correlated_views(n=20, d=5, seed=3)
        layer = ObservedMRDLayer.__new__(ObservedMRDLayer)
        Xc, _ = ObservedMRDLayer._init_X(layer, [Y1, Y2], 4, init='PCA_concat')
        self.assertEqual(Xc.shape, (20, 4))
        Xs, _ = ObservedMRDLayer._init_X(layer, [Y1, Y2], 4, init='PCA_single')
        self.assertEqual(Xs.shape, (20, 4))
        Xr, _ = ObservedMRDLayer._init_X(layer, [Y1, Y2], 4, init='random')
        self.assertEqual(Xr.shape, (20, 4))
        # Legacy alias
        Xp, _ = ObservedMRDLayer._init_X(layer, [Y1, Y2], 4, init='PCA')
        self.assertEqual(Xp.shape, (20, 4))


class MRDBackConstraintTest(unittest.TestCase):
    def test_back_constraint_constructs_and_optimizes(self):
        Y1, Y2 = _correlated_views(n=18, d=5, seed=4)
        q1, q2 = 3, 2
        # encoder_dims[0]: per-view MLP hidden widths; encoder_dims[1]: top layer
        m = deepgp.DeepGP(
            [[Y1.shape[1], Y2.shape[1]], q1, q2],
            [Y1, Y2],
            kernels=_mrd_kernels(q1, q2),
            num_inducing=6,
            back_constraint=True,
            encoder_dims=[[[4], [4]], [3]],
        )
        self.assertTrue(m._mrd)
        self.assertTrue(hasattr(m.layers[0].views[0], 'encoder'))
        ll = float(m.log_likelihood())
        self.assertTrue(np.isfinite(ll))
        m.optimize(max_iters=3, messages=False)
        self.assertTrue(np.isfinite(float(m.log_likelihood())))

    def test_mlp_dims_length_checked(self):
        Y1, Y2 = _correlated_views(n=10, d=4, seed=5)
        q1 = 3
        with self.assertRaises(AssertionError):
            deepgp.DeepGP(
                [[Y1.shape[1], Y2.shape[1]], q1],
                [Y1, Y2],
                kernels=_mrd_kernels(q1),
                num_inducing=4,
                back_constraint=True,
                encoder_dims=[[[3]]],  # only one view entry for two views
            )


if __name__ == '__main__':
    unittest.main()
