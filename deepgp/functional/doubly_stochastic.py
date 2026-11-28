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
                          reparam_sample, reparam_sample_backward,
                          whitened_expectation_grads)


def _check_layers(layers, X):
    width = X.shape[1]
    for i, l in enumerate(layers):
        if l.get('residual', False) and l['q_mu'].shape[1] != width:
            raise ValueError("layer %d: residual needs output width %d = input width, got %d"
                             % (i, l['q_mu'].shape[1], width))
        width = l['q_mu'].shape[1]
    return width


def doubly_stochastic_bound(layers, X, Y, noise_var, eps, jitter=1e-6, scale=1., kl_scale=1.,
                           x_var=None, eps_x=None, likelihood=None, views=None):
    """Bound and gradients for a stack of layers.

    :param eps: list over samples; each entry a list with one ``(N, width)`` standard
        normal array per hidden layer (``len(layers) - 1`` arrays; empty for one layer).
    :param x_var: optional ``(N, Q)`` variances of a Gaussian ``q(X) = N(X, diag(x_var))`` over
        the bottom inputs (the unsupervised case; ``X`` is then the mean). Each sample draws
        ``X + sqrt(x_var) * eps_x[s]`` (``eps_x``: one ``(N, Q)`` array per sample) and the bound
        subtracts ``scale * KL(q(X) || N(0, I))``.
    :param likelihood: optional :class:`~deepgp.functional.likelihoods.Likelihood`; ``None`` means
        Gaussian with ``noise_var`` (analytic, and ``grads['noise_var']`` is returned). With a
        likelihood object its expectations replace the Gaussian data term and ``noise_var`` is
        ignored (``grads['noise_var']`` is ``0.``).
    :param views: multiple output views (MRD). A list of ``dict(layer=..., Y=..., noise_var=...,
        likelihood=None)``, one whitened head per view. Then ``layers`` is the *shared trunk* only
        (possibly empty), ``eps`` has one array per trunk layer, and ``Y``, ``noise_var`` and
        ``likelihood`` are ignored. Every head reads the trunk output; the bound sums the views.
    :returns: ``(bound, grads)`` where ``grads`` has ``layers`` (list of dicts with
        ``q_mu``, ``q_chol``, ``kern``, ``Z``: the trunk, then the heads when ``views`` is given),
        ``noise_var`` (a list over views when ``views`` is given), ``X``, ``x_var`` (``0.`` unless
        given) and ``natgrad`` (``g_m (M, P)`` and ``g_S (P, M, M)``: gradients of the expected log
        likelihood, without the KL, with respect to the top layer's ``m`` and ``S``; see
        :func:`whitened_natgrad_step`). With ``views`` it is a list, one entry per head.
    """
    multi = views is not None
    if multi:
        trunk = list(layers)
        heads = [dict(v) for v in views]
        for v in heads:
            v.setdefault('likelihood', None)
    else:
        trunk = list(layers[:-1])
        heads = [dict(layer=layers[-1], Y=Y, noise_var=noise_var, likelihood=likelihood)]
    T = len(trunk)
    width = _check_layers(trunk, X)
    for v in heads:
        if v['Y'].shape[1] != v['layer']['q_mu'].shape[1]:
            raise ValueError("Y has %d columns but the head has %d outputs"
                             % (v['Y'].shape[1], v['layer']['q_mu'].shape[1]))
    N = X.shape[0]
    S = len(eps)
    all_layers = trunk + [v['layer'] for v in heads]
    layer_grads = [dict(q_mu=0., q_chol=0., kern=0., Z=0.) for _ in all_layers]
    gX, ell, g_xvar = 0., 0., 0.
    g_noise = [0. for _ in heads]
    if x_var is not None and (eps_x is None or len(eps_x) != len(eps)):
        raise ValueError("x_var needs one eps_x array per sample")
    nat = [[0., 0.] for _ in heads]

    for s_idx, sample in enumerate(eps):
        if len(sample) != T:
            raise ValueError("each sample needs %d eps arrays, got %d" % (T, len(sample)))
        inputs, caches, variances = [], [], []
        h = X if x_var is None else reparam_sample(X, x_var, eps_x[s_idx])
        for l, layer in enumerate(trunk):
            inputs.append(h)
            mu, var, cache = whitened_marginals(layer['kern'], layer['Z'].copy(), h.copy(),
                                                layer['q_mu'], layer['q_chol'], jitter)
            if layer.get('residual', False):
                mu = mu + h
            caches.append(cache); variances.append(var)
            h = reparam_sample(mu, var, sample[l])
        g_in = 0.
        for k, v in enumerate(heads):
            layer, Yv, nv, lik = v['layer'], v['Y'], v['noise_var'], v['likelihood']
            mu, var, cache = whitened_marginals(layer['kern'], layer['Z'].copy(), h.copy(),
                                                layer['q_mu'], layer['q_chol'], jitter)
            if lik is None:
                resid = Yv - mu
                ell += scale * np.sum(-0.5 * np.log(2 * np.pi * nv) - 0.5 * (resid ** 2 + var) / nv)
                g_noise[k] += scale * np.sum(-0.5 / nv + 0.5 * (resid ** 2 + var) / nv ** 2)
                g_mu = scale * resid / nv
                g_var = -0.5 * scale / nv * np.ones_like(var)
            else:
                e, gm, gv = lik.variational_expectations(Yv, mu, var)
                ell += scale * np.sum(e)
                g_mu, g_var = scale * gm, scale * gv
            g = whitened_marginals_backward(layer['kern'], layer['Z'].copy(), h.copy(),
                                            cache, g_mu, g_var)
            for key in ('q_mu', 'q_chol', 'kern', 'Z'):
                layer_grads[T + k][key] = layer_grads[T + k][key] + g[key] / S
            gm_, gS_ = whitened_expectation_grads(cache, g_mu, g_var)
            nat[k][0] = nat[k][0] + gm_ / S
            nat[k][1] = nat[k][1] + gS_ / S
            g_in = g_in + g['X']
        for l in range(T - 1, -1, -1):
            layer = trunk[l]
            g_mu, g_var = reparam_sample_backward(variances[l], sample[l], g_in)
            g = whitened_marginals_backward(layer['kern'], layer['Z'].copy(), inputs[l].copy(),
                                            caches[l], g_mu, g_var)
            for key in ('q_mu', 'q_chol', 'kern', 'Z'):
                layer_grads[l][key] = layer_grads[l][key] + g[key] / S
            g_in = g['X'] + (g_mu if layer.get('residual', False) else 0.)
        gX = gX + g_in / S
        if x_var is not None:
            g_xvar = g_xvar + reparam_sample_backward(x_var, eps_x[s_idx], g_in)[1] / S

    kl = 0.
    for l, layer in enumerate(all_layers):
        k, dm, dL = whitened_kl(layer['q_mu'], layer['q_chol'])
        kl += k
        layer_grads[l]['q_mu'] = layer_grads[l]['q_mu'] - kl_scale * dm
        layer_grads[l]['q_chol'] = layer_grads[l]['q_chol'] - kl_scale * dL

    bound = ell / S - kl_scale * kl
    if x_var is not None:
        kl_x = 0.5 * np.sum(X ** 2 + x_var - 1. - np.log(x_var))
        bound -= scale * kl_x
        gX = gX - scale * X
        g_xvar = g_xvar - scale * 0.5 * (1. - 1. / x_var)
    nats = [dict(g_m=m_, g_S=S_) for m_, S_ in nat]
    return bound, dict(layers=layer_grads, X=gX, x_var=g_xvar,
                       noise_var=[g / S for g in g_noise] if multi else g_noise[0] / S,
                       natgrad=nats if multi else nats[0])
