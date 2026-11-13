#!/usr/bin/env python
"""Section 4.1 (fig. 2): unsupervised recovery of a stacked-GP hierarchy.

Paper: Damianou & Lawrence, Deep Gaussian Processes, AISTATS 2013.

Generate a three-level stack (top latent → 2D intermediate → 10D observations),
fit a DeepGP with overcomplete ARD widths, and contrast against stacked PCA.
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
    ard_weights,
    dominant_dims,
    ensure_outdir,
    optimize_staged,
    print_ard,
    rbf_bias,
    sample_gp_prior,
    save_or_show,
)


def _pca_project(Y: np.ndarray, q: int) -> np.ndarray:
    try:
        from sklearn.decomposition import PCA

        return PCA(n_components=q).fit_transform(Y)
    except Exception:
        Yc = Y - Y.mean(0)
        U, S, _ = np.linalg.svd(Yc, full_matrices=False)
        q = min(q, U.shape[1])
        return U[:, :q] * S[:q]


def stacked_pca(Y: np.ndarray, dims: list[int]) -> list[np.ndarray]:
    """Greedy PCA stack returning latent matrices from bottom to top."""
    spaces = []
    cur = Y
    for q in dims:
        cur = _pca_project(cur, q)
        spaces.append(cur)
    return spaces


def generate_hierarchy(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Z_top, X_mid, Y_obs) for a 1→2→10 stacked GP."""
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    Z = np.linspace(-1.0, 1.0, n)[:, None]
    kern_mid = GPy.kern.RBF(1, lengthscale=0.4, variance=1.0)
    X = sample_gp_prior(Z, kern_mid, output_dim=2)
    X += 0.02 * rng.randn(*X.shape)
    kern_obs = GPy.kern.RBF(2, lengthscale=0.8, variance=1.0, ARD=False)
    Y = sample_gp_prior(X, kern_obs, output_dim=10)
    Y += 0.05 * rng.randn(*Y.shape)
    return Z, X, Y


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--n", type=int, default=80, help="Number of observations.")
    p.add_argument("--num-inducing", type=int, default=25)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outdir = ensure_outdir(args.outdir)
    n = 40 if args.quick else args.n
    q_mid, q_top = (4, 3) if args.quick else (5, 4)
    iters_fixed = 50 if args.quick else 200
    iters_free = 150 if args.quick else 1500
    num_inducing = 15 if args.quick else args.num_inducing

    Z, X_true, Y = generate_hierarchy(n, args.seed)
    print(f"Generated hierarchy: Y={Y.shape}, true mid={X_true.shape}, true top={Z.shape}")

    kernels = [rbf_bias(q_mid, ard=True), rbf_bias(q_top, ard=True)]
    m = deepgp.DeepGP(
        [Y.shape[1], q_mid, q_top],
        Y,
        kernels=kernels,
        num_inducing=num_inducing,
        back_constraint=False,
    )
    optimize_staged(m, iters_fixed, iters_free, messages=args.messages)
    print(f"DeepGP variational bound: {float(m.log_likelihood()):.3f}")
    print_ard(m)

    pca_spaces = stacked_pca(Y, [q_mid, q_top])

    if not args.no_plot or outdir:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        t = np.arange(n)
        axes[0].plot(t, Z[:, 0], "k-", label="top")
        axes[0].plot(t, X_true[:, 0], "C0-", label="mid0")
        axes[0].plot(t, X_true[:, 1], "C1-", label="mid1")
        axes[0].set_title("Ground truth")
        axes[0].legend(fontsize=8)

        for ax, layer_idx, title in (
            (axes[1], 0, "DeepGP layer 0 (near data)"),
            (axes[2], 1, "DeepGP layer 1 (top)"),
        ):
            Xmean = m.layers[layer_idx].X.mean
            dims = dominant_dims(ard_weights(m.layers[layer_idx].kern), k=min(2, Xmean.shape[1]))
            for d in dims:
                ax.plot(t, Xmean[:, d], label=f"dim {d}")
            ax.set_title(title)
            ax.legend(fontsize=8)

        fig.suptitle("Damianou & Lawrence 2013 — toy hierarchy (fig. 2 style)")
        fig.tight_layout()
        save_or_show(fig, outdir, "toy_hierarchy.png", args.no_plot)

        fig2, ax = plt.subplots(figsize=(5, 3))
        ax.plot(t, pca_spaces[0][:, 0], label="PCA mid 0")
        if pca_spaces[0].shape[1] > 1:
            ax.plot(t, pca_spaces[0][:, 1], label="PCA mid 1")
        ax.plot(t, pca_spaces[1][:, 0], label="PCA top 0")
        ax.set_title("Stacked PCA (greedy)")
        ax.legend(fontsize=8)
        fig2.tight_layout()
        save_or_show(fig2, outdir, "toy_hierarchy_pca.png", args.no_plot)

    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(main())
