"""GPc CIP-0035 verification oracle V8: the composite task (not run in CI).

A 10-D input is projected to two dimensions and passed through a nonlinearity
of the projection, so depth should help. The task generator is the one used by
GPc's JAX prototype and ``validation/gpc_vs_mlp.py`` (N=2000 train, 1000 test).
GPc's V8 gate is a three-layer inducing-point model with test RMSE <= 0.26 and
NLL <= 0 that beats the one-layer model; the prototype reached 0.229 / -0.25.

    python examples/composite_v8.py [iters] [seeds] [chi]
"""
import sys
import time

import numpy as np
import GPy

import deepgp


def task(seed=0):
    r = np.random.default_rng(seed)
    Bm = np.linalg.qr(r.normal(size=(10, 2)))[0]
    X = r.normal(size=(2000, 10))
    Xt = r.normal(size=(1000, 10))
    f = lambda x: np.sin(2 * (x @ Bm)[:, 0] + (x @ Bm)[:, 1] ** 2)
    n = 0.1
    y = f(X) + n * r.normal(size=len(X))
    yt = f(Xt) + n * r.normal(size=len(Xt))
    mu, sd = y.mean(), y.std()
    return X, ((y - mu) / sd)[:, None], Xt, ((yt - mu) / sd)[:, None]


def evaluate(model, Xt, yt, n_samples=50):
    mean, var = model.predict(Xt, n_samples=n_samples, seed=123)
    rmse = float(np.sqrt(np.mean((mean - yt) ** 2)))
    nll = float(-np.mean(-0.5 * np.log(2 * np.pi * var) - 0.5 * (yt - mean) ** 2 / var))
    return rmse, nll


def run(L, M=64, hidden=5, iters=3000, seed=0, natgrad_gamma=0., batch=256, chi=None):
    X, y, Xt, yt = task()
    nDims = [1] + [hidden] * (L - 1) + [10]
    m = deepgp.DeepGP(nDims, y, X, num_inducing=M, inference_method='doubly_stochastic',
                      seed=seed, noise_var=0.1, chi=chi)
    t0 = time.time()
    m.fit_minibatch(steps=iters, batch_size=batch, lr=0.01, seed=seed, natgrad_gamma=natgrad_gamma)
    sec = time.time() - t0
    m.set_batch(None)
    rmse, nll = evaluate(m, Xt, yt)
    return dict(L=L, M=M, chi=chi, rmse=rmse, nll=nll, elbo_per_n=m.log_likelihood() / len(X), sec=sec)


if __name__ == '__main__':
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    chi = float(sys.argv[3]) if len(sys.argv) > 3 else None
    for L in (2, 3) if chi is not None else (1, 2, 3):
        for s in range(seeds):
            print(run(L, iters=iters, seed=s, chi=chi), flush=True)
