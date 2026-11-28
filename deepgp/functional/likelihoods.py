# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""Observation likelihoods for the doubly stochastic top layer (pure, CIP-0003 Phase 4).

Each likelihood supplies ``log_density(y, f)`` and its derivative in ``f``. Given Gaussian
marginals ``q(f) = N(mu, var)`` from the top layer, :meth:`variational_expectations` returns
the elementwise expected log likelihood ``E_q[log p(y | f)]`` and its derivatives with respect
to ``mu`` and ``var``. Following GPc (``ndlutil::gaussHermiteExpect``) the expectation is a
20-point Gauss-Hermite rule, and the returned derivatives are those of the quadrature itself,
so they match finite differences of the returned value to round-off.

The likelihoods match GPc's: ``Probit`` (labels in {-1, +1}), ``Poisson`` (log link) and
``StudentT`` (variance and degrees of freedom held fixed). ``Gaussian`` is exact and is
included so that the generic path can be checked against the analytic one. ``log_concave``
says whether the natural-gradient step is safe (GPc CIP-0035: the step needs a log-concave
likelihood, otherwise the implied precision can be indefinite).

Not shipped yet: softmax and robust-max (they couple outputs) and learning likelihood
parameters.
"""

import numpy as np
from scipy.special import gammaln, log_ndtr, ndtr

_NODES, _WEIGHTS = np.polynomial.hermite.hermgauss(20)
_WEIGHTS = _WEIGHTS / np.sqrt(np.pi)


class Likelihood(object):
    """Base class: subclasses define ``log_density`` and ``d_log_density``."""
    log_concave = True

    def log_density(self, y, f):
        raise NotImplementedError

    def d_log_density(self, y, f):
        raise NotImplementedError

    def check(self, y):
        """Raise ``ValueError`` if ``y`` is not valid data for this likelihood."""

    def variational_expectations(self, y, mu, var):
        """``(ell, g_mu, g_var)``, each shaped like ``mu``, by Gauss-Hermite quadrature."""
        y, mu, var = (np.asarray(a, float) for a in (y, mu, var))
        if np.any(var < 0.):
            raise ValueError("var must be non-negative")
        scale = np.sqrt(2. * var)[..., None]
        f = mu[..., None] + scale * _NODES
        yy = y[..., None]
        ell = np.sum(_WEIGHTS * self.log_density(yy, f), axis=-1)
        d = _WEIGHTS * self.d_log_density(yy, f)
        g_mu = np.sum(d, axis=-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            g_var = np.sum(np.where(scale > 0., d * _NODES / scale, 0.), axis=-1)
        return ell, g_mu, g_var

    def predictive_density(self, y, mu, var):
        """``E_q[p(y | f)]`` elementwise, by Gauss-Hermite quadrature."""
        y, mu, var = (np.asarray(a, float) for a in (y, mu, var))
        f = mu[..., None] + np.sqrt(2. * var)[..., None] * _NODES
        return np.sum(_WEIGHTS * np.exp(self.log_density(y[..., None], f)), axis=-1)

    def predictive_mean(self, mu, var):
        """``E[y]`` under ``q(f) = N(mu, var)``."""
        raise NotImplementedError


class Gaussian(Likelihood):
    """``y = f + N(0, noise_var)``: exact expectations (no quadrature)."""

    def __init__(self, noise_var=1.):
        self.noise_var = float(noise_var)

    def log_density(self, y, f):
        return -0.5 * np.log(2 * np.pi * self.noise_var) - 0.5 * (y - f) ** 2 / self.noise_var

    def d_log_density(self, y, f):
        return (y - f) / self.noise_var

    def variational_expectations(self, y, mu, var):
        r = np.asarray(y, float) - mu
        s2 = self.noise_var
        ell = -0.5 * np.log(2 * np.pi * s2) - 0.5 * (r ** 2 + var) / s2
        return ell, r / s2, np.full(np.shape(mu), -0.5 / s2)

    def predictive_density(self, y, mu, var):
        v = var + self.noise_var
        return np.exp(-0.5 * (np.asarray(y, float) - mu) ** 2 / v) / np.sqrt(2 * np.pi * v)

    def predictive_mean(self, mu, var):
        return mu


class Probit(Likelihood):
    """``p(y | f) = Phi(y (f + bias))`` with ``y`` in {-1, +1}.

    As in GPc the bias shifts the latent function; it is held fixed here. Use
    :meth:`from_labels` for GPc's initial value, ``Phi^{-1}`` of the fraction of positive labels.
    """

    def __init__(self, bias=0.):
        self.bias = float(bias)

    @classmethod
    def from_labels(cls, y):
        from scipy.special import ndtri
        return cls(bias=float(ndtri(np.mean(np.asarray(y) == 1.))))

    def check(self, y):
        if not np.all(np.isin(y, (-1., 1.))):
            raise ValueError("Probit expects labels in {-1, +1}")

    def log_density(self, y, f):
        return log_ndtr(y * (f + self.bias))

    def d_log_density(self, y, f):
        z = y * (f + self.bias)
        return y * np.exp(-0.5 * z ** 2 - 0.5 * np.log(2 * np.pi) - log_ndtr(z))

    def predictive_mean(self, mu, var):
        """``P(y = +1)`` (exact for the probit link)."""
        return ndtr((mu + self.bias) / np.sqrt(1. + var))


class Poisson(Likelihood):
    """Counts with rate ``exp(f)``."""

    def check(self, y):
        if np.any(y < 0) or not np.allclose(y, np.round(y)):
            raise ValueError("Poisson expects non-negative integer counts")

    def log_density(self, y, f):
        return -np.exp(f) + y * f - gammaln(y + 1.)

    def d_log_density(self, y, f):
        return y - np.exp(f)

    def predictive_mean(self, mu, var):
        return np.exp(mu + 0.5 * var)


class StudentT(Likelihood):
    """Heavy-tailed regression, GPc parameterisation: scale ``variance`` and ``deg_free``.

    Not log-concave, so natural-gradient steps are refused.
    """
    log_concave = False

    def __init__(self, variance=1., deg_free=4.):
        if variance <= 0. or deg_free <= 0.:
            raise ValueError("variance and deg_free must be positive")
        self.variance, self.deg_free = float(variance), float(deg_free)

    def log_density(self, y, f):
        v, s = self.deg_free, self.variance
        e2 = (y - f) ** 2
        return (gammaln(0.5 * (v + 1.)) - gammaln(0.5 * v) - 0.5 * np.log(s * v * np.pi)
                - 0.5 * (v + 1.) * np.log1p(e2 / (s * v)))

    def d_log_density(self, y, f):
        v, s = self.deg_free, self.variance
        e = y - f
        return (v + 1.) * e / (s * v + e ** 2)

    def predictive_mean(self, mu, var):
        return mu
