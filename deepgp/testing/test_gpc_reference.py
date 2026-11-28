"""Cross-check of the doubly stochastic bound against GPc CIP-0035 reference values.

The golden file ``gpc_reference.json`` is produced by
``gpc_reference_generate.py`` with GPc's compiled ``DeepGP``; CI does not
depend on GPc.  One layer is deterministic and must agree to rounding; deeper
models are Monte Carlo estimates and must agree within sampling error.
"""
import json
import os
import unittest

import numpy as np
import GPy

from deepgp.functional import doubly_stochastic_bound, whitened_kl, whitened_marginals

_PATH = os.path.join(os.path.dirname(__file__), 'gpc_reference.json')


def _layers(case):
    L = case['num_layers']
    out = []
    for l, d in enumerate(case['layers']):
        Z = np.array(d['Z'])
        kern = GPy.kern.RBF(Z.shape[1],
                            lengthscale=np.exp(-0.5 * d['log_inverse_width']),
                            variance=np.exp(d['log_variance']))
        out.append(dict(kern=kern, Z=Z, q_mu=np.array(d['q_mu']),
                        q_chol=np.array(d['q_chol']), residual=l < L - 1))
    return out, 1. / np.exp(case['layers'][-1]['log_beta'])


class TestGPcReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(_PATH) as f:
            cls.data = json.load(f)
        cls.X = np.array(cls.data['X'])
        cls.Y = np.array(cls.data['Y'])

    def _case(self, name):
        return next(c for c in self.data['cases'] if c['name'] == name)

    def _eps(self, L, S, seed=0):
        rng = np.random.RandomState(seed)
        N, Q = self.X.shape
        return [[rng.randn(N, Q) for _ in range(L - 1)] for _ in range(S)]

    def test_one_layer_elbo_exact(self):
        case = self._case('one_layer')
        layers, noise = _layers(case)
        bound, _ = doubly_stochastic_bound(layers, self.X, self.Y, noise,
                                           self._eps(1, 1))
        self.assertAlmostEqual(bound, case['elbo'], places=4)

    def test_one_layer_kl_exact(self):
        case = self._case('one_layer')
        layers, _ = _layers(case)
        kl = whitened_kl(layers[0]['q_mu'], layers[0]['q_chol'])[0]
        self.assertAlmostEqual(kl, case['layer_kl'][0], places=6)

    def test_one_layer_prediction_exact(self):
        case = self._case('one_layer')
        layers, noise = _layers(case)
        l = layers[0]
        mu, var, _ = whitened_marginals(l['kern'], l['Z'],
                                        np.array(self.data['Xtest']),
                                        l['q_mu'], l['q_chol'])
        np.testing.assert_allclose(mu, case['pred_mean'], atol=1e-5)
        np.testing.assert_allclose(var[:, None] + noise if var.ndim == 1
                                   else var + noise,
                                   case['pred_var'], atol=1e-5)

    def _check_stochastic(self, name, S=800):
        case = self._case(name)
        layers, noise = _layers(case)
        # Our estimate has its own sampling error; compare with combined s.e.
        vals = []
        for seed in range(4):
            b, _ = doubly_stochastic_bound(layers, self.X, self.Y, noise,
                                           self._eps(case['num_layers'], S // 4,
                                                     seed))
            vals.append(b)
        ours = float(np.mean(vals))
        # per-seed estimates are means over S/4 samples; their spread gives s.e.
        se_ours = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        se = np.hypot(se_ours, case['elbo_stderr'])
        self.assertLess(abs(ours - case['elbo_mean']), 4 * se + 1e-3,
                        '%s: ours %.4f gpc %.4f (se %.4f)'
                        % (name, ours, case['elbo_mean'], se))

    def test_two_layer_elbo_statistical(self):
        self._check_stochastic('two_layer')

    def test_three_layer_elbo_statistical(self):
        self._check_stochastic('three_layer')

    # --- non-Gaussian top layers (GPc defaults: StudentT variance 1, 5 degrees of freedom) ---
    def _lik(self, name, Y):
        from deepgp.functional import likelihoods as lk
        return dict(probit=lk.Probit.from_labels(np.asarray(Y)), poisson=lk.Poisson(),
                    student_t=lk.StudentT(1., 5.))[name]

    def _lcase(self, name):
        return next(c for c in self.data['likelihood_cases'] if c['name'] == name)

    def test_non_gaussian_one_layer_elbo_exact(self):
        for lik in ('probit', 'poisson', 'student_t'):
            case = self._lcase(lik + '_1')
            layers, _ = _layers(case)
            Y = np.array(case['Y'])
            bound, _ = doubly_stochastic_bound(layers, self.X, Y, 1., self._eps(1, 1),
                                               likelihood=self._lik(lik, case['Y']))
            self.assertAlmostEqual(bound, case['elbo'], places=4, msg=lik)

    def test_non_gaussian_two_layer_elbo_statistical(self):
        for lik in ('probit', 'poisson', 'student_t'):
            case = self._lcase(lik + '_2')
            layers, _ = _layers(case)
            Y = np.array(case['Y'])
            vals = [doubly_stochastic_bound(layers, self.X, Y, 1., self._eps(2, 200, seed),
                                            likelihood=self._lik(lik, case['Y']))[0] for seed in range(4)]
            se = np.hypot(np.std(vals, ddof=1) / 2., case['elbo_stderr'])
            self.assertLess(abs(np.mean(vals) - case['elbo_mean']), 4 * se + 1e-3,
                            '%s: ours %.4f gpc %.4f (se %.4f)' % (lik, np.mean(vals), case['elbo_mean'], se))

    def test_layer_kl_all_cases(self):
        for case in self.data['cases']:
            layers, _ = _layers(case)
            for l, ref in zip(layers, case['layer_kl']):
                kl = whitened_kl(l['q_mu'], l['q_chol'])[0]
                self.assertAlmostEqual(kl, ref, places=5)


if __name__ == '__main__':
    unittest.main()
