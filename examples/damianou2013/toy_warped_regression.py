#!/usr/bin/env python
"""§4.1 / fig. 3 — warped regression (Matlab ``demToyRegressionSimple.m``).

Data: ``hierGpsNEW`` in ``demToyDynamicsCreateData.m`` (RBF+Linear warp → RBF).
Defaults follow the Matlab simple demo (Ntoy=150, D=15, Ntr=25, H=2, Q=6, K=10).
Use ``--paper`` for the AISTATS text settings (N=120, D=10).
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
    save_or_show,
)
from _matlab_data import generate_hier_gps_new, train_test_split_indices


def fit_gp(Xtr, Ytr, num_inducing, messages):
    kern = GPy.kern.RBF(Xtr.shape[1]) + GPy.kern.Bias(Xtr.shape[1])
    m = GPy.models.SparseGPRegression(Xtr, Ytr, kernel=kern, num_inducing=num_inducing)
    m.Gaussian_noise.variance = Ytr.var() * 0.01
    m.Gaussian_noise.variance.fix()
    m.optimize(max_iters=100, messages=messages)
    m.Gaussian_noise.variance.unfix()
    m.optimize(max_iters=400, messages=messages)
    return m


def fit_dgp(Xtr, Ytr, q, num_inducing, iters_fixed, iters_free, messages):
    # H=2 hidden layers between observed X and Y (Matlab simple / paper)
    dims = [Ytr.shape[1], q, q, Xtr.shape[1]]
    kernels = [
        rbf_bias(q, ard=True),
        rbf_bias(q, ard=True),
        # Top mapping sees inputs; Matlab uses lin+white+bias ("dynamics") —
        # approximate with RBF+Bias on the observed X layer.
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument(
        "--paper",
        action="store_true",
        help="Use AISTATS text sizes (N=120, D=10) instead of Matlab simple (150, 15).",
    )
    p.add_argument("--repeats", type=int, default=10)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outdir = ensure_outdir(args.outdir)

    if args.paper:
        n_toy, d_toy = 120, 10
    else:
        n_toy, d_toy = 150, 15
    n_tr = 25
    q = 3 if args.quick else 6
    num_inducing = 8 if args.quick else 10
    repeats = 2 if args.quick else args.repeats
    iters_fixed = 40 if args.quick else 500
    iters_free = 120 if args.quick else 2000

    print(
        f"Config: Ntoy={n_toy} D={d_toy} Ntr={n_tr} Q={q} K={num_inducing} "
        f"repeats={repeats} ({'paper' if args.paper else 'matlab-simple'})"
    )

    gp_mses, dgp_mses = [], []
    for r in range(repeats):
        seed = args.seed + r
        X, H, Y = generate_hier_gps_new(n_toy=n_toy, d_toy=d_toy, seed=seed)
        ind_tr, ind_ts = train_test_split_indices(n_toy, n_tr, seed=seed + 1000)
        Xtr, Ytr = X[ind_tr], Y[ind_tr]
        Xte, Yte = X[ind_ts], Y[ind_ts]

        m_gp = fit_gp(Xtr, Ytr, num_inducing, messages=False)
        m_dgp = fit_dgp(
            Xtr, Ytr, q, num_inducing, iters_fixed, iters_free, messages=args.messages
        )
        gp_mses.append(mse(m_gp.predict(Xte)[0], Yte))
        dgp_mses.append(mse(m_dgp.predict(Xte)[0], Yte))
        print(f"Repeat {r + 1}/{repeats}: MSE GP={gp_mses[-1]:.4f}  DeepGP={dgp_mses[-1]:.4f}")

    print(
        f"Mean MSE (±std): GP {np.mean(gp_mses):.4f}±{np.std(gp_mses):.4f}  "
        f"DeepGP {np.mean(dgp_mses):.4f}±{np.std(dgp_mses):.4f}"
    )

    if not args.no_plot or outdir:
        import matplotlib.pyplot as plt

        X, H, Y = generate_hier_gps_new(n_toy=n_toy, d_toy=d_toy, seed=args.seed)
        fig, axes = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
        axes[0].plot(X[:, 0], H[:, 0], "k-")
        axes[0].set_ylabel("hidden warp Y1")
        axes[0].set_title("hierGpsNEW generative process")
        axes[1].plot(X[:, 0], Y[:, 0], label="Y[:,0]")
        axes[1].plot(X[:, 0], Y[:, 1], label="Y[:,1]")
        axes[1].set_xlabel("X (scaled)")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        save_or_show(fig, outdir, "toy_warped_process.png", args.no_plot)

        fig2, ax = plt.subplots(figsize=(5, 3.5))
        xs = np.arange(1, repeats + 1)
        ax.plot(xs, gp_mses, "o-", label="sparse GP")
        ax.plot(xs, dgp_mses, "s-", label="DeepGP")
        ax.set_xlabel("experiment #")
        ax.set_ylabel("test MSE")
        ax.set_title("Fig. 3(b) style / demToyRegressionSimple")
        ax.legend()
        fig2.tight_layout()
        save_or_show(fig2, outdir, "toy_warped_mse.png", args.no_plot)

    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(main())
