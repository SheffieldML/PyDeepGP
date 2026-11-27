#!/usr/bin/env python
"""§4.1 / fig. 2 — unsupervised hierarchy (Matlab ``demToyUnsupervised.m``).

Loads ``hgplvmSampleDataTr1.mat`` (top ``X2`` → intermediates ``XA``,``XB`` →
leaves ``YA``,``YB``), fits a two-view DeepGP with latent widths Q={6,4}
matching the Matlab demo, and compares against stacked PCA and Isomap.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import deepgp

from _common import (
    add_common_args,
    ard_weights,
    dominant_dims,
    ensure_outdir,
    optimize_staged,
    rbf_bias,
    save_or_show,
)
from _matlab_data import load_toy_hierarchy


def stacked_pca(Ya: np.ndarray, Yb: np.ndarray, q1: int, q2: int):
    def pca(Y, q):
        Yc = Y - Y.mean(0)
        U, S, _ = np.linalg.svd(Yc, full_matrices=False)
        q = min(q, U.shape[1])
        return U[:, :q] * S[:q]

    xa, xb = pca(Ya, q1), pca(Yb, q1)
    x2 = pca(np.hstack([xa, xb]), q2)
    return xa, xb, x2


def stacked_isomap(Ya: np.ndarray, Yb: np.ndarray, q1: int, q2: int, n_neighbors: int = 5):
    try:
        from sklearn.manifold import Isomap
    except ImportError as exc:
        raise SystemExit("stacked Isomap requires scikit-learn") from exc

    xa = Isomap(n_components=min(q1, Ya.shape[0] - 1), n_neighbors=n_neighbors).fit_transform(Ya)
    xb = Isomap(n_components=min(q1, Yb.shape[0] - 1), n_neighbors=n_neighbors).fit_transform(Yb)
    x2 = Isomap(
        n_components=min(q2, xa.shape[0] - 1), n_neighbors=n_neighbors
    ).fit_transform(np.hstack([xa, xb]))
    return xa, xb, x2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--num-inducing", type=int, default=25)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outdir = ensure_outdir(args.outdir)

    data = load_toy_hierarchy()
    YA, YB = data["YA"], data["YB"]
    X2, XA, XB = data["X2"], data["XA"], data["XB"]
    print(
        f"Loaded hgplvmSampleDataTr1: YA={YA.shape} YB={YB.shape} "
        f"XA={XA.shape} XB={XB.shape} X2={X2.shape}"
    )

    # Matlab demToyUnsupervised: Q = {6,4}
    q_shared, q_top = (4, 3) if args.quick else (6, 4)
    num_inducing = 15 if args.quick else args.num_inducing
    iters_fixed = 50 if args.quick else 200
    iters_free = 150 if args.quick else 2000

    kernels = [
        [rbf_bias(q_shared), rbf_bias(q_shared)],
        rbf_bias(q_top),
    ]
    m = deepgp.DeepGP(
        [[YA.shape[1], YB.shape[1]], q_shared, q_top],
        [YA, YB],
        kernels=kernels,
        num_inducing=num_inducing,
        back_constraint=False,
        inits="PCA",
    )
    optimize_staged(m, iters_fixed, iters_free, messages=args.messages)
    print(f"DeepGP variational bound: {float(m.log_likelihood()):.3f}")
    for i, view in enumerate(m.layers[0].views):
        print(f"View {i} ARD: {np.array2string(ard_weights(view.kern), precision=3)}")
    print(f"Top ARD: {np.array2string(ard_weights(m.layers[-1].kern), precision=3)}")

    pca_xa, pca_xb, pca_x2 = stacked_pca(YA, YB, q_shared, q_top)
    try:
        iso_xa, iso_xb, iso_x2 = stacked_isomap(YA, YB, min(2, q_shared), min(2, q_top))
        have_iso = True
    except SystemExit:
        have_iso = False
        print("sklearn not available — skipping Isomap comparison")

    if not args.no_plot or outdir:
        import matplotlib.pyplot as plt

        t = np.arange(YA.shape[0])
        fig, axes = plt.subplots(3, 2, figsize=(9, 8))
        axes[0, 0].plot(X2)
        axes[0, 0].set_title("True top X2")
        axes[0, 1].axis("off")
        axes[1, 0].plot(XA)
        axes[1, 0].set_title("True XA")
        axes[1, 1].plot(XB)
        axes[1, 1].set_title("True XB")
        axes[2, 0].plot(YA)
        axes[2, 0].set_title("YA")
        axes[2, 1].plot(YB)
        axes[2, 1].set_title("YB")
        fig.suptitle("Matlab hgplvmSampleDataTr1 ground truth")
        fig.tight_layout()
        save_or_show(fig, outdir, "toy_hierarchy_truth.png", args.no_plot)

        Xshare = np.asarray(m.layers[0].X.mean)
        Xtop = np.asarray(m.layers[-1].X.mean)
        d_top = dominant_dims(ard_weights(m.layers[-1].kern), k=min(2, Xtop.shape[1]))
        w0 = ard_weights(m.layers[0].views[0].kern)
        w1 = ard_weights(m.layers[0].views[1].kern)
        d_a = dominant_dims(w0, k=min(2, Xshare.shape[1]))
        d_b = dominant_dims(w1, k=min(2, Xshare.shape[1]))

        fig2, axes = plt.subplots(2, 2, figsize=(8, 7))
        axes[0, 0].plot(Xtop[:, d_top])
        axes[0, 0].set_title("DeepGP top (dominant dims)")
        axes[0, 1].axis("off")
        axes[1, 0].plot(Xshare[:, d_a])
        axes[1, 0].set_title("DeepGP shared (view0 ARD dims)")
        axes[1, 1].plot(Xshare[:, d_b])
        axes[1, 1].set_title("DeepGP shared (view1 ARD dims)")
        fig2.suptitle("demToyUnsupervised-style DeepGP recovery")
        fig2.tight_layout()
        save_or_show(fig2, outdir, "toy_hierarchy_deepgp.png", args.no_plot)

        fig3, axes = plt.subplots(2, 2, figsize=(8, 7))
        axes[0, 0].plot(pca_x2[:, : min(2, pca_x2.shape[1])])
        axes[0, 0].set_title("Stacked PCA top")
        axes[1, 0].plot(pca_xa[:, : min(2, pca_xa.shape[1])])
        axes[1, 0].set_title("Stacked PCA XA")
        axes[1, 1].plot(pca_xb[:, : min(2, pca_xb.shape[1])])
        axes[1, 1].set_title("Stacked PCA XB")
        axes[0, 1].axis("off")
        fig3.suptitle("Stacked PCA")
        fig3.tight_layout()
        save_or_show(fig3, outdir, "toy_hierarchy_pca.png", args.no_plot)

        if have_iso:
            fig4, axes = plt.subplots(2, 2, figsize=(8, 7))
            axes[0, 0].plot(iso_x2)
            axes[0, 0].set_title("Stacked Isomap top")
            axes[1, 0].plot(iso_xa)
            axes[1, 0].set_title("Stacked Isomap XA")
            axes[1, 1].plot(iso_xb)
            axes[1, 1].set_title("Stacked Isomap XB")
            axes[0, 1].axis("off")
            fig4.suptitle("Stacked Isomap")
            fig4.tight_layout()
            save_or_show(fig4, outdir, "toy_hierarchy_isomap.png", args.no_plot)

    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(main())
