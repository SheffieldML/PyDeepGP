# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""Doubly stochastic (uncollapsed) deep GP as a GPy model (CIP-0003 Phase 4).

A thin shell over :mod:`deepgp.functional.doubly_stochastic`: it holds the
parameters (kernels, inducing inputs, whitened ``q(u)``, noise) as GPy objects and
calls the pure functions. ``X=None`` gives the unsupervised model with latent ``q(X)``.

``nDims`` follows ``DeepGP``: ``[D_y, ..., D_x]``, output first. ``kernels`` follows
the same order, so ``kernels[0]`` is the layer that produces ``Y``.

Randomness is explicit. The model holds a frozen set of standard normal draws
``eps`` so that the objective is deterministic (``checkgrad`` and ``optimize``
work). Call :meth:`resample` to draw new ones from the seeded generator, and
:meth:`fit_minibatch` to do so every step.
"""

import numpy as np
import GPy
from GPy import Model
from GPy.core.parameterization import Param, Parameterized
from GPy.core.parameterization.transformations import Logexp
from GPy.util import choleskies
from GPy.util.pca import PCA

from ..functional import (doubly_stochastic_bound, whitened_marginals, reparam_sample,
                          whitened_natgrad_step)


class WhitenedLayer(Parameterized):
    """Parameters of one whitened sparse-GP layer."""

    def __init__(self, kern, Z, q_mu, q_chol, residual=False, name='whitened_layer'):
        super(WhitenedLayer, self).__init__(name=name)
        self.kern = kern
        self.Z = Param('Z', Z)
        self.q_mu = Param('q_u_mean', q_mu)
        self.q_chol_flat = Param('q_u_chol', choleskies.triang_to_flat(q_chol))
        self.residual = residual
        self.link_parameters(self.kern, self.Z, self.q_mu, self.q_chol_flat)

    @property
    def q_chol(self):
        return choleskies.flat_to_triang(self.q_chol_flat.values)

    def as_dict(self):
        return dict(kern=self.kern, Z=self.Z.values, q_mu=self.q_mu.values,
                    q_chol=self.q_chol, residual=self.residual)


class DoublyStochasticDeepGP(Model):
    """Deep GP trained with doubly stochastic variational inference.

    With ``X=None`` the model is unsupervised: the bottom inputs are latent with
    ``q(X) = N(X_mean, diag(X_var))``, initialised by PCA of ``Y`` (``nDims[-1]`` is the latent
    width), sampled by reparameterisation, and regularised by ``KL(q(X) || N(0, I))``.

    :param nDims: ``[D_y, ..., D_x]``; hidden widths in between.
    :param Y: ``(N, D_y)``; :param X: ``(N, D_x)``, or ``None`` for the unsupervised model.
    :param kernels: optional list in ``nDims`` order (``kernels[0]`` produces ``Y``).
    :param residual: identity mean function on hidden layers where widths allow.
    :param n_samples: Monte Carlo samples per evaluation.
    :param batch_size: default minibatch size for :meth:`fit_minibatch` (``None`` = all).
    :param seed: seed of the model's random generator.
    """

    def __init__(self, nDims, Y, X=None, num_inducing=20, kernels=None, noise_var=0.1,
                 n_samples=1, residual=True, batch_size=None, seed=0, jitter=1e-6,
                 q_sqrt_hidden=1e-2, q_sqrt_x=0.5, name='doubly_stochastic_deepgp'):
        super(DoublyStochasticDeepGP, self).__init__(name=name)
        dims = list(nDims)[::-1]                       # input side first
        self.Y_all = np.asarray(Y, float)
        self.unsupervised = X is None
        if dims[-1] != self.Y_all.shape[1]:
            raise ValueError("nDims %s does not match Y (%d)" % (list(nDims), self.Y_all.shape[1]))
        if not self.unsupervised:
            if dims[0] != X.shape[1]:
                raise ValueError("nDims %s does not match X (%d)" % (list(nDims), X.shape[1]))
            if X.shape[0] != Y.shape[0]:
                raise ValueError("X and Y must have the same number of rows")
        self.rng = np.random.RandomState(seed)
        self.N = self.Y_all.shape[0]
        if self.unsupervised:
            if len(dims) < 2 or dims[0] > self.Y_all.shape[1]:
                raise ValueError("latent width %d must be at most the number of outputs %d"
                                 % (dims[0], self.Y_all.shape[1]))
            x0 = PCA(self.Y_all).project(self.Y_all, dims[0])
            x0 = (x0 - x0.mean(0)) / (x0.std(0) + 1e-12)
            self.X_all = x0
        else:
            self.X_all = np.asarray(X, float)
        self.n_samples, self.batch_size, self.jitter = n_samples, batch_size, jitter
        self.idx = np.arange(self.N)
        L = len(dims) - 1
        kernels = None if kernels is None else list(kernels)[::-1]

        # Hidden representations initialised by PCA of Y, as in DeepGP.
        H = [self.X_all]
        for w in dims[1:-1]:
            p = PCA(self.Y_all)
            h = p.project(self.Y_all, min(w, self.Y_all.shape[1]))
            full = self.rng.randn(self.N, w) * 0.1
            full[:, :h.shape[1]] = h
            H.append((full - full.mean(0)) / (full.std(0) + 1e-12))

        self.layers = []
        for i in range(L):
            Q, P = dims[i], dims[i + 1]
            M = num_inducing if np.isscalar(num_inducing) else num_inducing[i]
            M = min(M, self.N)
            kern = kernels[i] if kernels is not None else GPy.kern.RBF(Q, ARD=True)
            Z = H[i][self.rng.permutation(self.N)[:M]].copy()
            hidden = i < L - 1
            s = q_sqrt_hidden if hidden else 1.
            q_chol = np.stack([np.eye(M) * s for _ in range(P)])
            self.layers.append(WhitenedLayer(kern, Z, np.zeros((M, P)), q_chol,
                                             residual=bool(residual and hidden and Q == P),
                                             name='layer_%d' % i))
        self.noise_variance = Param('noise_variance', noise_var, Logexp())
        if self.unsupervised:
            self.X_mean = Param('X_mean', self.X_all.copy())
            self.X_var = Param('X_var', np.full(self.X_all.shape, q_sqrt_x ** 2), Logexp())
            self.link_parameters(self.X_mean, self.X_var)
        self.link_parameters(*(self.layers + [self.noise_variance]))
        self.resample()

    # --- randomness -------------------------------------------------------
    def resample(self, seed=None):
        """Draw a fresh frozen set of ``eps`` (optionally reseeding the generator)."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        n = len(self.idx)
        widths = [l.q_mu.shape[1] for l in self.layers[:-1]]
        self.eps = [[self.rng.randn(n, w) for w in widths] for _ in range(self.n_samples)]
        if self.unsupervised:
            self.eps_x = [self.rng.randn(n, self.X_all.shape[1]) for _ in range(self.n_samples)]
        self.parameters_changed()

    def set_batch(self, idx=None):
        """Use rows ``idx`` of the data (``None`` = all); draws new ``eps`` of the right size."""
        self.idx = np.arange(self.N) if idx is None else np.asarray(idx)
        self.resample()

    # --- GPy model interface ---------------------------------------------
    def parameters_changed(self):
        if not hasattr(self, 'eps'):
            return
        Y = self.Y_all[self.idx]
        layers = [l.as_dict() for l in self.layers]
        extra = {}
        if self.unsupervised:
            X = self.X_mean.values[self.idx]
            extra = dict(x_var=self.X_var.values[self.idx], eps_x=self.eps_x)
        else:
            X = self.X_all[self.idx]
        bound, g = doubly_stochastic_bound(layers, X, Y, float(self.noise_variance.values[0]), self.eps,
                                           jitter=self.jitter, scale=self.N / float(len(self.idx)),
                                           **extra)
        if self.unsupervised:
            gm, gv = np.zeros_like(self.X_all), np.zeros_like(self.X_all)
            gm[self.idx], gv[self.idx] = g['X'], g['x_var']
            self.X_mean.gradient, self.X_var.gradient = gm, gv
        self._log_marginal_likelihood = bound
        self._natgrad = g['natgrad']
        for layer, gl in zip(self.layers, g['layers']):
            layer.kern.gradient = gl['kern']
            layer.Z.gradient = gl['Z']
            layer.q_mu.gradient = gl['q_mu']
            layer.q_chol_flat.gradient = choleskies.triang_to_flat(gl['q_chol'])
        self.noise_variance.gradient = g['noise_var']

    def log_likelihood(self):
        return self._log_marginal_likelihood

    # --- training ---------------------------------------------------------
    def fit_minibatch(self, steps=200, batch_size=None, lr=0.01, beta1=0.9, beta2=0.999,
                      seed=None, callback=None, natgrad_gamma=0.):
        """Adam on minibatches; fresh batch and ``eps`` every step, from the model's generator.

        Parameters are updated in GPy's unconstrained space. At the end the full batch
        is restored and ``eps`` redrawn. Returns the list of per-step bounds.

        With ``natgrad_gamma > 0`` the top layer's ``q(u)`` takes a natural-gradient step of
        that size (see ``whitened_natgrad_step``) instead of an Adam step, and everything
        else stays on Adam: the hybrid scheme of Salimbeni, Eleftheriadis and Hensman (2018).
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        B = batch_size or self.batch_size or self.N
        theta = self.optimizer_array.copy()
        m, v, history = np.zeros_like(theta), np.zeros_like(theta), []
        for t in range(1, steps + 1):
            self.idx = self.rng.permutation(self.N)[:B]
            self.resample()
            obj_grad = -self._log_likelihood_gradients()      # gradient of minimised objective
            if natgrad_gamma > 0.:
                top = self.layers[-1]
                new_mu, new_chol = whitened_natgrad_step(
                    top.q_mu.values, top.q_chol, self._natgrad['g_m'], self._natgrad['g_S'],
                    natgrad_gamma)
                new_flat = choleskies.triang_to_flat(new_chol)
            m = beta1 * m + (1 - beta1) * obj_grad
            v = beta2 * v + (1 - beta2) * obj_grad ** 2
            theta = theta - lr * (m / (1 - beta1 ** t)) / (np.sqrt(v / (1 - beta2 ** t)) + 1e-8)
            self.optimizer_array = theta
            if natgrad_gamma > 0.:
                top.q_mu[:] = new_mu
                top.q_chol_flat[:] = new_flat
                theta = self.optimizer_array.copy()
            history.append(self._log_marginal_likelihood)
            if callback is not None:
                callback(t, history[-1])
        self.set_batch(None)
        return history

    # --- prediction --------------------------------------------------------
    def predict(self, Xnew, n_samples=None, seed=None):
        """Mean and variance of ``y`` at ``Xnew`` from a Monte Carlo mixture over hidden layers.

        Variance includes the noise. Uses its own generator (``seed``) so the training
        ``eps`` are not disturbed.
        """
        S = n_samples or max(self.n_samples, 20)
        rng = np.random.RandomState(0 if seed is None else seed)
        means, vars_ = [], []
        for _ in range(S):
            h = np.asarray(Xnew, float)
            for l, layer in enumerate(self.layers):
                d = layer.as_dict()
                mu, var, _ = whitened_marginals(d['kern'], d['Z'].copy(), h.copy(),
                                                d['q_mu'], d['q_chol'], self.jitter)
                if d['residual']:
                    mu = mu + h
                if l < len(self.layers) - 1:
                    h = reparam_sample(mu, var, rng.randn(*mu.shape))
            means.append(mu); vars_.append(var)
        means, vars_ = np.array(means), np.array(vars_)
        mean = means.mean(0)
        var = (vars_ + means ** 2).mean(0) - mean ** 2 + float(self.noise_variance.values[0])
        return mean, var
