#!/usr/bin/env python
"""Section 4.1 (fig. 3): warped toy regression — DeepGP vs sparse GP.

Paper: Damianou & Lawrence, Deep Gaussian Processes, AISTATS 2013.

Stack Linear+RBF → RBF to make a non-stationary 10D regression problem from a
1D input. Train on 25 points; compare MSE of a standard sparse GP against a
DeepGP that is one hidden layer deeper than the generative process.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import GPy
import deepgp

from _common import (
    add_common_args,
    ensure_outdir,
    mse,
    optimize_staged,
    rbf_bias,
    sample_gp_prior,
    save_or_show,
)


def generate_warped(
    n: int = 120,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, H_hidden, Y) for the paper's warped process."""
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    X = np.linspace(-1.0, 1.0, n)[:, None]
    kern_warp = GPy.kern.Linear(1) + GPy.kern.RBF(1, lengthscale=0.3, variance=1.0)
    H = sample_gp_prior(X, kern_warp, output_dim=1)
    H += 0.01 * rng.randn(*H.shape)
    kern_obs = GPy.kern.RBF(1, lengthscale=0.25, variance=1.0)
    Y = sample_gp_prior(H, kern_obs, output_dim=10)
    Y += 0.05 * rng.randn(*Y.shape)
    return X, H, Y


def train_test_split(
    X: np.ndarray,
    Y: np.ndarray,
    n_train: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    idx = rng.permutation(X.shape[0])
    tr, te = np.sort(idx[:n_train]), np.sort(idx[n_train:])
    return X[tr], Y[tr], X[te], Y[te]


def fit_deepgp(
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    q1: int,
    q2: int,
    num_inducing: int,
    iters_fixed: int,
    iters_free: int,
    messages: bool,
) -> deepgp.DeepGP:
    # Two hidden layers between X and Y (paper: one more than the generative depth)
    dims = [Ytr.shape[1], q1, q2, Xtr.shape[1]]
    kernels = [
        rbf_bias(q1, ard=True),
        rbf_bias(q2, ard=True),
        rbf_bias(Xtr.shape[1], ard=False),
    ]
    m = deepgp.DeepGP(
        dims,
        Ytr,
        X=Xtr,
        kernels=kernels,
        num_inducing=num_inducing,
        back_constraint=False,
    )
    optimize_staged(m, iters_fixed, iters_free, messages=messages)
    return m


def fit_gp(
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    num_inducing: int,
    messages: bool,
) -> GPy.core.Model:
    kern = GPy.kern.RBF(Xtr.shape[1]) + GPy.kern.Bias(Xtr.shape[1])
    m = GPy.models.SparseGPRegression(Xtr, Ytr, kernel=kern, num_inducing=num_inducing)
    m.Gaussian_noise.variance = Ytr.var() * 0.01
    m.Gaussian_noise.variance.fix()
    m.optimize(max_iters=100, messages=messages)
    m.Gaussian_noise.variance.unfix()
    m.optimize(max_iters=400, messages=messages)
    return m


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--repeats", type=int, default=10, help="Paper uses 10 repeats.")
    p.add_argument("--n-train", type=int, default=25)
    p.add_argument("--num-inducing", type=int, default=20)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outdir = ensure_outdir(args.outdir)

    repeats = 2 if args.quick else args.repeats
    n_train = 20 if args.quick else args.n_train
    q1, q2 = (3, 2) if args.quick else (4, 3)
    iters_fixed = 40 if args.quick else 200
    iters_free = 120 if args.quick else 1200
    num_inducing = 12 if args.quick else args.num_inducing

    gp_mses = []
    dgp_mses = []

    for r in range(repeats):
        seed = args.seed + r
        X, H, Y = generate_warped(n=120, seed=seed)
        Xtr, Ytr, Xte, Yte = train_test_split(X, Y, n_train=n_train, seed=seed + 1000)

        m_gp = fit_gp(Xtr, Ytr, num_inducing=num_inducing, messages=False)
        m_dgp = fit_deepgp(
            Xtr,
            Ytr,
            q1=q1,
            q2=q2,
            num_inducing=num_inducing,
            iters_fixed=iters_fixed,
            iters_free=iters_free,
            messages=args.messages,
        )

        mse_gp = mse(m_gp.predict(Xte)[0], Yte)
        mse_dgp = mse(m_dgp.predict(Xte)[0], Yte)
        gp_mses.append(mse_gp)
        dgp_mses.append(mse_dgp)
        print(f"Repeat {r + 1}/{repeats}: MSE GP={mse_gp:.4f}  DeepGP={mse_dgp:.4f}")

    print(
        f"Mean MSE (±std): GP {np.mean(gp_mses):.4f}±{np.std(gp_mses):.4f}  "
        f"DeepGP {np.mean(dgp_mses):.4f}±{np.std(dgp_mses):.4f}"
    )

    if not args.no_plot or outdir:
        import matplotlib.pyplot as plt

        # Illustrate one generative draw
        X, H, Y = generate_warped(n=120, seed=args.seed)
        fig, axes = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
        axes[0].plot(X[:, 0], H[:, 0], "k-")
        axes[0].set_ylabel("hidden warp")
        axes[0].set_title("Warped process (one draw)")
        axes[1].plot(X[:, 0], Y[:, 0], label="Y[:,0]")
        axes[1].plot(X[:, 0], Y[:, 1], label="Y[:,1]")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("outputs")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        save_or_show(fig, outdir, "toy_warped_process.png", args.no_plot)

        fig2, ax = plt.subplots(figsize=(5, 3.5))
        xs = np.arange(1, repeats + 1)
        ax.plot(xs, gp_mses, "o-", label="GP")
        ax.plot(xs, dgp_mses, "s-", label="DeepGP")
        ax.set_xlabel("experiment #")
        ax.set_ylabel("MSE")
        ax.set_title("Damianou & Lawrence 2013 — fig. 3(b) style")
        ax.legend()
        fig2.tight_layout()
        save_or_show(fig2, outdir, "toy_warped_mse.png", args.no_plot)

    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(main())
