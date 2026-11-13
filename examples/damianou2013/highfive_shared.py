#!/usr/bin/env python
"""§4.2 / figs. 4–5 — CMU high-five shared MRD (Matlab ``demHighFiveDemonstration.m``).

Loads ``demHighFiveHgplvm1.mat`` (subjects A/B, 78×62 each) and fits a two-view
DeepGP. Compares ARD / shared latents to the stored hgplvm reference spaces in
the same mat file when present.
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
    init_noise,
    rbf_bias,
    save_or_show,
    unfix_noise,
)
from _matlab_data import load_highfive


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outdir = ensure_outdir(args.outdir)

    data = load_highfive()
    YA, YB = data["YA"].copy(), data["YB"].copy()
    YA = (YA - YA.mean(0)) / (YA.std() + 1e-12)
    YB = (YB - YB.mean(0)) / (YB.std() + 1e-12)
    print(f"Loaded demHighFiveHgplvm1: YA={YA.shape} YB={YB.shape}")

    # Two-level hierarchy as in the paper / demHighFiveDemonstration
    q1, q2 = (4, 3) if args.quick else (6, 4)
    num_inducing = 15 if args.quick else 30
    iters_fixed = 40 if args.quick else 200
    iters_free = 80 if args.quick else 1500

    m = deepgp.DeepGP(
        [[YA.shape[1], YB.shape[1]], q1, q2],
        [YA, YB],
        kernels=[[rbf_bias(q1), rbf_bias(q1)], rbf_bias(q2)],
        num_inducing=num_inducing,
        back_constraint=False,
        inits="PCA",
    )
    init_noise(m, fix=True)
    m.optimize(max_iters=iters_fixed, messages=args.messages)
    unfix_noise(m)
    m.optimize(max_iters=iters_free, messages=args.messages)
    print(f"MRD DeepGP variational bound: {float(m.log_likelihood()):.3f}")

    for i, view in enumerate(m.layers[0].views):
        w = ard_weights(view.kern)
        print(f"View {i} ARD (1/lengthscale): {np.array2string(w, precision=3)}")
    print(f"Top ARD: {np.array2string(ard_weights(m.layers[-1].kern), precision=3)}")

    if not args.no_plot or outdir:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        for i, view in enumerate(m.layers[0].views):
            w = ard_weights(view.kern)
            axes[i].bar(np.arange(1, len(w) + 1), w)
            axes[i].set_title(f"view {i} ARD (fY)")
            axes[i].set_xlabel("shared dim")
        w_top = ard_weights(m.layers[-1].kern)
        axes[2].bar(np.arange(1, len(w_top) + 1), w_top)
        axes[2].set_title("top ARD (fX)")
        fig.suptitle("High-five MRD ARD (fig. 4 style)")
        fig.tight_layout()
        save_or_show(fig, outdir, "highfive_ard.png", args.no_plot)

        Xshare = np.asarray(m.layers[0].X.mean)
        Xtop = np.asarray(m.layers[-1].X.mean)
        d_share = dominant_dims(ard_weights(m.layers[0].views[0].kern), k=2)
        d_top = dominant_dims(ard_weights(m.layers[-1].kern), k=2)

        fig2, axes = plt.subplots(1, 2, figsize=(8, 3.5))
        axes[0].plot(Xshare[:, d_share[0]], Xshare[:, d_share[1]], ".-")
        axes[0].set_title("DeepGP shared (dominant)")
        axes[1].plot(Xtop[:, d_top[0]], Xtop[:, d_top[1]], ".-")
        axes[1].set_title("DeepGP top (dominant)")
        fig2.suptitle("Latent projections (fig. 5 style)")
        fig2.tight_layout()
        save_or_show(fig2, outdir, "highfive_latents.png", args.no_plot)

        if "Xparent" in data:
            fig3, axes = plt.subplots(1, 3, figsize=(10, 3.2))
            axes[0].plot(data["Xparent"][:, 0], data["Xparent"][:, 1], ".-")
            axes[0].set_title("hgplvm Xparent (ref)")
            axes[1].plot(data["Xchild1"][:, 0], data["Xchild1"][:, 1], ".-")
            axes[1].set_title("hgplvm Xchild1 (ref)")
            axes[2].plot(data["Xchild2"][:, 0], data["Xchild2"][:, 1], ".-")
            axes[2].set_title("hgplvm Xchild2 (ref)")
            fig3.suptitle("Reference latents from demHighFiveHgplvm1.mat")
            fig3.tight_layout()
            save_or_show(fig3, outdir, "highfive_hgplvm_ref.png", args.no_plot)

    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(main())
