"""ARD scale inspection, prune/restore and initialiser names (CIP-0003 Phase 3)."""
import unittest

import numpy as np
import GPy

import deepgp
from deepgp.diagnostics import ard_scales, prune_model, restore_model, show_scales
from deepgp.util.util import validate_init


def _model(lengthscales=(1., 1e4), seed=3):
    rng = np.random.RandomState(seed)
    X = rng.randn(12, 1)
    Y = np.hstack([np.sin(X), np.cos(X), X]) + 0.05 * rng.randn(12, 3)
    np.random.seed(seed)
    k = [GPy.kern.RBF(2, ARD=True, lengthscale=list(lengthscales))]
    return deepgp.DeepGP([3, 2], Y, kernels=k, num_inducing=5, back_constraint=False)


class ScalesTest(unittest.TestCase):
    def test_scales_identify_irrelevant_dimension(self):
        s = ard_scales(_model())
        self.assertEqual(list(s), ['layer_0'])
        self.assertAlmostEqual(s['layer_0'][0], 1.0)
        self.assertLess(s['layer_0'][1], 1e-6)

    def test_unnormalised_scales(self):
        s = ard_scales(_model(lengthscales=(2., 4.)), normalize=False)['layer_0']
        np.testing.assert_allclose(s, [0.25, 1. / 16])

    def test_mrd_scales_per_view(self):
        rng = np.random.RandomState(1)
        X = rng.randn(10, 2)
        Y1, Y2 = X @ rng.randn(2, 3), X @ rng.randn(2, 3)
        m = deepgp.DeepGP([[3, 3], 2], [Y1, Y2], num_inducing=4, back_constraint=False,
                          kernels=[[GPy.kern.RBF(2, ARD=True), GPy.kern.RBF(2, ARD=True)]])
        self.assertEqual(list(ard_scales(m)), ['layer_0/view_0', 'layer_0/view_1'])

    def test_plot_helper(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig = show_scales(_model())
        self.assertEqual(len(fig.axes), 1)
        plt.close(fig)


class PruneTest(unittest.TestCase):
    def test_prune_restore_round_trip(self):
        m = _model()
        before, params = m.log_likelihood(), m.param_array.copy()
        report = prune_model(m)
        np.testing.assert_array_equal(report['layer_0'], [1])
        # An irrelevant dimension: the objective barely moves.
        self.assertLess(abs(m.log_likelihood() - before), 1e-3 * (1 + abs(before)))
        self.assertTrue(m.layers[0].kern.lengthscale[1:].is_fixed)
        restore_model(m)
        self.assertFalse(m.layers[0].kern.lengthscale[1:].is_fixed)
        np.testing.assert_allclose(m.param_array, params)
        self.assertAlmostEqual(m.log_likelihood(), before, places=10)

    def test_keeps_at_least_one_dimension(self):
        m = _model(lengthscales=(1., 1.))
        prune_model(m, threshold=10.)  # would remove everything
        self.assertEqual(len(m.layers[0].kern.lengthscale.values), 2)
        np.testing.assert_allclose(m.layers[0].kern.lengthscale.values, [1., 1.])

    def test_double_prune_and_bare_restore_fail_clearly(self):
        m = _model()
        with self.assertRaises(RuntimeError):
            restore_model(m)
        prune_model(m)
        with self.assertRaises(RuntimeError):
            prune_model(m)
        restore_model(m)


class InitTest(unittest.TestCase):
    def test_names(self):
        for name in ('PCA', 'rand', 'random', 'randomProjection', 'bgplvm'):
            self.assertEqual(validate_init(name), name)
        self.assertEqual(validate_init('PCA_single', mrd=True), 'PCA_single')
        with self.assertRaises(ValueError):
            validate_init('PCA_single')
        with self.assertRaises(ValueError):
            validate_init('pca-ish')

    def test_unknown_init_rejected_by_model(self):
        Y = np.random.randn(8, 3)
        with self.assertRaises(ValueError):
            deepgp.DeepGP([3, 2], Y, inits='nonsense', num_inducing=4, back_constraint=False)

    def test_each_init_builds_and_default_is_pca(self):
        np.random.seed(0)
        Y = np.random.randn(10, 3)
        for name in ('PCA', 'random', 'randomProjection'):
            m = deepgp.DeepGP([3, 2], Y, inits=name, num_inducing=4, back_constraint=False)
            self.assertTrue(np.isfinite(m.log_likelihood()))
        import inspect
        self.assertEqual(inspect.signature(deepgp.DeepGP.__init__).parameters['inits'].default, 'PCA')


class ToyExampleTest(unittest.TestCase):
    def test_optimisation_reveals_relevant_dimension(self):
        """Data generated from one latent signal, three latent dimensions offered."""
        rng = np.random.RandomState(0)
        t = np.linspace(-2, 2, 30)[:, None]
        Y = np.hstack([np.sin(2 * t), np.cos(t), t ** 2 / 2.]) + 0.02 * rng.randn(30, 3)
        np.random.seed(0)
        m = deepgp.DeepGP([3, 3], Y, num_inducing=8, back_constraint=False)
        m.optimize(max_iters=300, messages=False)
        s = ard_scales(m)['layer_0']
        order = np.sort(s)[::-1]
        self.assertAlmostEqual(order[0], 1.0)
        self.assertLess(order[-1], 0.5 * order[0])
        report = prune_model(m, threshold=0.5 * order[0] + 1e-9)
        restore_model(m)
        self.assertIn('layer_0', report)


if __name__ == '__main__':
    unittest.main()
