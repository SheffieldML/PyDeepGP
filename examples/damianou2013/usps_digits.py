#!/usr/bin/env python
"""§4.3 / figs. 6–8 — USPS {0,1,6} (Matlab ``demUsps.m`` / ``demDigitsDemonstration.m``).

``vercat2``: 50 examples of digits 0, 6, 1. Depth-5 demo uses
Q = {15, 12, 10, 8, 6} and K = 80 inducing points (experimentNo 19 comment).
Reports variational bound and leave-one-out 1-NN error in the top latent space.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np

import deepgp

from _common import (
    add_common_args,
    ard_weights,
    dominant_dims,
    ensure_outdir,
    nearest_neighbour_error,
    optimize_staged,
    print_ard,
    rbf_bias,
    save_or_show,
)
from _matlab_data import USPS_DEPTH_Q, load_usps_vercat2


def depth_dimensions(depth: int, data_dim: int) -> List[int]:
    if depth not in USPS_DEPTH_Q:
        raise ValueError(f"depth must be in {sorted(USPS_DEPTH_Q)}")
    return [data_dim] + list(USPS_DEPTH_Q[depth])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=None,
        help="Default: 1..5 (or 1..2 with --quick).",
    )
    p.add_argument("--per-class", type=int, default=50)
    p.add_argument("--num-inducing", type=int, default=80)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outdir = ensure_outdir(args.outdir)

    depths = args.depths
    if depths is None:
        depths = [1, 2] if args.quick else [1, 2, 3, 4, 5]

    per_class = 20 if args.quick else args.per_class
    # Matlab uses K=80; keep a reduced inducing set only for --quick
    num_inducing = 25 if args.quick else args.num_inducing
    iters_fixed = 40 if args.quick else 500
    iters_free = 100 if args.quick else 3000

    Y, labels = load_usps_vercat2(per_class=per_class)
    Y = Y - Y.mean(0)
    Y = Y / (Y.std() + 1e-12)
    print(f"USPS vercat2-style: Y={Y.shape}, classes={sorted(set(labels.tolist()))}")

    results = []
    models = {}
    for depth in depths:
        print(f"\n=== depth {depth}  dims={depth_dimensions(depth, Y.shape[1])} ===")
        dims = depth_dimensions(depth, Y.shape[1])
        # For quick runs, shrink Q to keep optimisation tractable
        if args.quick:
            dims = [Y.shape[1]] + [max(2, q // 2) for q in dims[1:]]
        m = deepgp.DeepGP(
            dims,
            Y,
            kernels=[rbf_bias(q) for q in dims[1:]],
            num_inducing=num_inducing,
            back_constraint=False,
            inits="PCA",
        )
        optimize_staged(m, iters_fixed, iters_free, messages=args.messages)
        bound = float(m.log_likelihood())
        Xtop = np.asarray(m.layers[-1].X.mean)
        mistakes, rate = nearest_neighbour_error(Xtop, labels)
        print(f"bound={bound:.3f}  1-NN mistakes={mistakes}/{Y.shape[0]} ({rate:.3%})")
        print_ard(m)
        results.append((depth, bound, mistakes, rate))
        models[depth] = m

    print("\nSummary (Matlab demDigitsDemonstration metrics)")
    print("depth  bound      1-NN errors")
    for depth, bound, mistakes, rate in results:
        print(f"{depth:5d}  {bound:9.3f}  {mistakes:3d} ({rate:.3%})")

    deepest = max(depths)
    m = models[deepest]
    if not args.no_plot or outdir:
        import matplotlib.pyplot as plt

        Xtop = np.asarray(m.layers[-1].X.mean)
        dims_plot = dominant_dims(ard_weights(m.layers[-1].kern), k=2)
        colors = {0: "C0", 1: "C1", 6: "C2"}
        markers = {0: "o", 1: "s", 6: "^"}
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        for d in (0, 1, 6):
            mask = labels == d
            ax.scatter(
                Xtop[mask, dims_plot[0]],
                Xtop[mask, dims_plot[1]],
                c=colors[d],
                marker=markers[d],
                s=28,
                label=str(d),
            )
        ax.set_title(f"USPS top latent (depth {deepest})")
        ax.legend()
        fig.tight_layout()
        save_or_show(fig, outdir, "usps_top_latent.png", args.no_plot)

        fig2, axes = plt.subplots(1, len(m.layers), figsize=(3.0 * len(m.layers), 3))
        if len(m.layers) == 1:
            axes = [axes]
        for i, layer in enumerate(m.layers):
            w = ard_weights(layer.kern)
            axes[i].bar(np.arange(1, len(w) + 1), w)
            axes[i].set_title(f"layer {i}")
        fig2.suptitle("ARD (fig. 6 style)")
        fig2.tight_layout()
        save_or_show(fig2, outdir, "usps_ard.png", args.no_plot)

        fig3, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot([r[0] for r in results], [r[1] for r in results], "o-", label="bound")
        ax.set_xlabel("depth")
        ax.set_ylabel("variational bound")
        ax2 = ax.twinx()
        ax2.plot(
            [r[0] for r in results],
            [r[2] for r in results],
            "s--",
            color="C1",
            label="1-NN mistakes",
        )
        ax2.set_ylabel("1-NN mistakes")
        ax.set_title("Depth vs bound / 1-NN")
        fig3.tight_layout()
        save_or_show(fig3, outdir, "usps_depth.png", args.no_plot)

    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(main())
