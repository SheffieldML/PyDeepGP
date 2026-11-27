# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""Doubly stochastic variational bound for a stack of whitened sparse GPs (pure).

Following Salimbeni & Deisenroth (2017). Inputs ``X`` are propagated through the
hidden layers by reparameterised samples (``f = mu + sqrt(var) * eps``); the top
layer supplies Gaussian marginals, for which the expected log likelihood is
analytic. The bound is the sample average of

    scale * sum_{n,p} E_q[log N(y_np | f_np, noise_var)]  -  kl_scale * sum_l KL_l,

where ``scale = N / batch_size`` for minibatches. ``eps`` is supplied by the caller
so results are reproducible and gradients testable.

A layer is a dict with keys ``kern``, ``Z``, ``q_mu``, ``q_chol`` and optionally
``residual`` (bool). With ``residual`` and equal input/output width the layer adds
its input to its mean (an identity mean function), which Salimbeni & Deisenroth
recommend for hidden layers.
"""

import numpy as np

from .uncollapsed import (whitened_marginals, whitened_marginals_backward, whitened_kl,
                          reparam_sample, reparam_sample_backward)


def _check_layers(layers, X):
    width = X.shape[1]
    for i, l in enumerate(layers):
        if l.get('residual', False) and l['q_mu'].shape[1] != width:
            raise ValueError("layer %d: residual needs output width %d = input width, got %d"
                             % (i, l['q_mu'].shape[1], width))
        width = l['q_mu'].shape[1]
    return width


def doubly_stochastic_bound(layers, X, Y, noise_var, eps, jitter=1e-6, scale=1., kl_scale=1.):
    """Bound and gradients for a stack of layers.

    :param eps: list over samples; each entry a list with one ``(N, width)`` standard
        normal array per hidden layer (``len(layers) - 1`` arrays; empty for one layer).
    :returns: ``(bound, grads)`` where ``grads`` has ``layers`` (list of dicts with
        ``q_mu``, ``q_chol``, ``kern``, ``Z``), ``noise_var`` and ``X``.
    """
    L = len(layers)
    out_width = _check_layers(layers, X)
    if Y.shape[1] != out_width:
        raise ValueError("Y has %d columns but the top layer has %d outputs" % (Y.shape[1], out_width))
    N = X.shape[0]
    S = len(eps)
    layer_grads = [dict(q_mu=0., q_chol=0., kern=0., Z=0.) for _ in range(L)]
    gX, g_noise, ell = 0., 0., 0.

    for sample in eps:
        if len(sample) != L - 1:
            raise ValueError("each sample needs %d eps arrays, got %d" % (L - 1, len(sample)))
        inputs, caches, variances = [], [], []
        h = X
        for l, layer in enumerate(layers):
            inputs.append(h)
            mu, var, cache = whitened_marginals(layer['kern'], layer['Z'].copy(), h.copy(),
                                                layer['q_mu'], layer['q_chol'], jitter)
            if layer.get('residual', False):
                mu = mu + h
            caches.append(cache); variances.append(var)
            if l < L - 1:
                h = reparam_sample(mu, var, sample[l])
        resid = Y - mu
        ell += scale * np.sum(-0.5 * np.log(2 * np.pi * noise_var) - 0.5 * (resid ** 2 + var) / noise_var)
        g_noise += scale * np.sum(-0.5 / noise_var + 0.5 * (resid ** 2 + var) / noise_var ** 2)

        g_mu = scale * resid / noise_var
        g_var = -0.5 * scale / noise_var * np.ones_like(var)
        for l in range(L - 1, -1, -1):
            layer = layers[l]
            g = whitened_marginals_backward(layer['kern'], layer['Z'].copy(), inputs[l].copy(),
                                            caches[l], g_mu, g_var)
            for k in ('q_mu', 'q_chol', 'kern', 'Z'):
                layer_grads[l][k] = layer_grads[l][k] + g[k] / S
            g_in = g['X'] + (g_mu if layer.get('residual', False) else 0.)
            if l > 0:
                g_mu, g_var = reparam_sample_backward(variances[l - 1], sample[l - 1], g_in)
            else:
                gX = gX + g_in / S

    kl = 0.
    for l, layer in enumerate(layers):
        k, dm, dL = whitened_kl(layer['q_mu'], layer['q_chol'])
        kl += k
        layer_grads[l]['q_mu'] = layer_grads[l]['q_mu'] - kl_scale * dm
        layer_grads[l]['q_chol'] = layer_grads[l]['q_chol'] - kl_scale * dL

    bound = ell / S - kl_scale * kl
    return bound, dict(layers=layer_grads, noise_var=g_noise / S, X=gX)
