"""Collapsed (VarDTC / Bayesian GP-LVM) versus doubly stochastic deep GPs on oil and CMU mocap.

Not run in CI (needs ``pods`` and downloads data).

Oil (100 points, 12-D, 3 classes): unsupervised ``[12, 5, 2]`` stacks. We report the
leave-one-out 1-nearest-neighbour class error in the 2-D top latent space and the
training reconstruction RMSE.

CMU subject 35, walk (62-D joint angles): train on every other frame of one walk, hold out the
frames in between. The held-out latent is the mean of the neighbouring training latents, pushed
through the model; we report held-out RMSE. This tests how smooth and informative the
learned top latent is, not dynamics.

    python examples/compare_collapsed_vs_doubly_stochastic.py [oil|cmu] [seed]
"""
import sys
import time

import numpy as np

import deepgp


def standardise(Y):
    mu, sd = Y.mean(0), Y.std(0)
    sd[sd < 1e-8] = 1.
    return (Y - mu) / sd, mu, sd


def loo_1nn_error(X, labels):
    D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    return float(np.mean(labels[D.argmin(1)] != labels))


def collapsed_once(Y, nDims, M, iters, seed, noise_factor):
    np.random.seed(seed)
    m = deepgp.DeepGP(nDims, Y, num_inducing=M)
    for i, layer in enumerate(m.layers):
        out_var = layer.Y.var() if i == 0 else layer.Y.mean.var()
        layer.Gaussian_noise.variance = out_var * noise_factor
    # Restart L-BFGS until it stops improving the bound (or the iteration budget is spent), so the
    # baseline is converged rather than cut off by a single call.
    best, used = -np.inf, 0
    while used < iters:
        m.optimize(max_iters=min(1000, iters - used), messages=False)
        used += 1000
        if m.log_likelihood() < best + 1e-3 * abs(best):
            break
        best = m.log_likelihood()
    m.iters_used = min(used, iters)
    return m


INITS = [(0, 0.01), (1000, 0.01), (0, 0.1), (1000, 0.1)]     # (seed offset, noise / output variance)


def collapsed(Y, nDims, M, iters, seed, inits=INITS):
    """Best of several initialisations, chosen by the variational bound (never by the metric).

    ``m.all_bounds`` lists the bound of every initialisation.
    """
    t0 = time.time()
    models = [collapsed_once(Y, nDims, M, iters, seed + off, nf) for off, nf in inits]
    bounds = [m.log_likelihood() for m in models]
    m = models[int(np.argmax(bounds))]
    m.all_bounds, m.all_models = bounds, models
    return m, time.time() - t0


def doubly_stochastic(Y, nDims, M, iters, seed, natgrad_gamma=0., chi=None):
    m = deepgp.DeepGP(nDims, Y, num_inducing=M, inference_method='doubly_stochastic',
                      seed=seed, noise_var=0.05, n_samples=1, chi=chi)
    t0 = time.time()
    m.fit_minibatch(steps=iters, batch_size=min(len(Y), 128), lr=0.01, seed=seed,
                    natgrad_gamma=natgrad_gamma)
    m.set_batch(None)
    return m, time.time() - t0


VARIANTS = [(0., None), (0.1, None)]      # chi is a no-op for two-layer stacks


def label(gamma, chi):
    return 'doubly stochastic' + (' + natgrad' if gamma else '') + (' + chi=%g' % chi if chi else '')


def top_latent(m):
    return m.X_mean.values if hasattr(m, 'X_mean') else np.asarray(m.layers[-1].X.mean)


def oil(seed=0, iters=(5000, 3000)):
    import pods
    d = pods.datasets.oil_100()
    Y, _, _ = standardise(d['X'])
    labels = d['Y'].argmax(1)
    nDims, M = [Y.shape[1], 5, 2], 30
    rows = []
    mc, tc = collapsed(Y, nDims, M, iters[0], seed)
    rows.append(('collapsed (%d iters)' % mc.iters_used, loo_1nn_error(top_latent(mc), labels),
                 float(np.sqrt(np.mean((mc.predict(top_latent(mc))[0] - Y) ** 2))), tc))
    for k, mk in enumerate(mc.all_models):
        print('  collapsed init %s: bound %.1f, 1-NN %.2f, RMSE %.3f' % (
            INITS[k], mc.all_bounds[k], loo_1nn_error(top_latent(mk), labels),
            float(np.sqrt(np.mean((mk.predict(top_latent(mk))[0] - Y) ** 2)))), flush=True)
    for gamma, chi in VARIANTS:
        md, td = doubly_stochastic(Y, nDims, M, iters[1], seed, natgrad_gamma=gamma, chi=chi)
        mean, _ = md.predict(top_latent(md), n_samples=20, seed=0)
        rows.append((label(gamma, chi),
                     loo_1nn_error(top_latent(md), labels),
                     float(np.sqrt(np.mean((mean - Y) ** 2))), td))
    return rows


def cmu(seed=0, iters=(5000, 3000)):
    import pods
    d = pods.datasets.cmu_mocap('35', ['01'], [], sample_every=1)
    Y, _, _ = standardise(d['Y'])
    Y = Y[:(len(Y) // 2) * 2]
    train, test = Y[0::2], Y[1::2]
    nDims, M = [Y.shape[1], 5, 3], 25
    rows = []

    def held_out(model, X, predict):
        Xmid = 0.5 * (X[:-1] + X[1:])               # between consecutive training frames
        mean = predict(Xmid)
        return float(np.sqrt(np.mean((mean - test[:-1]) ** 2)))

    mc, tc = collapsed(train, nDims, M, iters[0], seed)
    Xc = top_latent(mc)
    rows.append(('collapsed (%d iters)' % mc.iters_used, held_out(mc, Xc, lambda X: mc.predict(X)[0]), tc))
    for k, mk in enumerate(mc.all_models):
        print('  collapsed init %s: bound %.1f, held-out RMSE %.3f' % (
            INITS[k], mc.all_bounds[k], held_out(mk, top_latent(mk), lambda X: mk.predict(X)[0])),
            flush=True)
    for gamma, chi in VARIANTS:
        md, td = doubly_stochastic(train, nDims, M, iters[1], seed, natgrad_gamma=gamma, chi=chi)
        Xd = top_latent(md)
        rows.append((label(gamma, chi),
                     held_out(md, Xd, lambda X: md.predict(X, n_samples=20, seed=0)[0]), td))
    rows.append(('mean of train frames', float(np.sqrt(np.mean(test ** 2))), 0.))
    return rows


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'oil'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    for row in (oil if which == 'oil' else cmu)(seed):
        print(row, flush=True)
