#!/usr/bin/env python
"""Section 4.2 (figs. 4–5): CMU high-five — shared / MRD hierarchy.

Paper: Damianou & Lawrence, Deep Gaussian Processes, AISTATS 2013.

Two observation views share a parent latent layer (conditionally independent
given the shared features). With CIP-0002, this uses the maintained
``ObservedMRDLayer`` path.

Real CMU high-five MOCAP (two subjects, ~78 frames × 62 dims) is used when
``pods`` can supply a suitable sequence; otherwise the script falls back to
synthetic correlated views so the shared-hierarchy fit still runs.
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
    ensure_outdir,
    init_noise,
    print_ard,
    rbf_bias,
    save_or_show,
    unfix_noise,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Force synthetic correlated views (skip pods MOCAP).",
    )
    return p


def load_two_views(seed: int, synthetic: bool):
    """Return (Y1, Y2, source_label)."""
    if not synthetic:
        try:
            import pods

            if hasattr(pods.datasets, "cmu_mocap_35_walk_jog"):
                data = pods.datasets.cmu_mocap_35_walk_jog()
                Y = np.asarray(data["Y"], dtype=float)
                n = min(80, Y.shape[0])
                Y = Y[:n]
                mid = Y.shape[1] // 2
                print(
                    "Note: using walk/jog feature split as a two-view stand-in; "
                    "paper §4.2 used the CMU high-five pair of subjects."
                )
                return Y[:, :mid], Y[:, mid:], "pods:cmu_mocap_35_walk_jog"
        except Exception as exc:
            print(f"pods MOCAP unavailable ({exc}); using synthetic views.")

    rng = np.random.RandomState(seed)
    n, q, d = 60, 3, 20
    X = rng.randn(n, q)
    Y1 = X @ rng.randn(q, d) + 0.05 * rng.randn(n, d)
    Y2 = X @ rng.randn(q, d) + 0.05 * rng.randn(n, d)
    return Y1, Y2, "synthetic"


def fit_mrd(Y1, Y2, quick: bool, messages: bool) -> deepgp.DeepGP:
    q1, q2 = (4, 3) if quick else (6, 4)
    num_inducing = 15 if quick else 30
    iters_fixed = 40 if quick else 200
    iters_free = 80 if quick else 800
    kernels = [
        [rbf_bias(q1, ard=True), rbf_bias(q1, ard=True)],
        rbf_bias(q2, ard=True),
    ]
    m = deepgp.DeepGP(
        [[Y1.shape[1], Y2.shape[1]], q1, q2],
        [Y1, Y2],
        kernels=kernels,
        num_inducing=num_inducing,
        back_constraint=False,
    )
    init_noise(m, fix=True)
    m.optimize(max_iters=iters_fixed, messages=messages)
    unfix_noise(m)
    m.optimize(max_iters=iters_free, messages=messages)
    return m


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outdir = ensure_outdir(args.outdir)

    Y1, Y2, source = load_two_views(args.seed, synthetic=args.synthetic or args.quick)
    Y1 = (Y1 - Y1.mean(0)) / (Y1.std() + 1e-12)
    Y2 = (Y2 - Y2.mean(0)) / (Y2.std() + 1e-12)
    print(f"Views: {Y1.shape}, {Y2.shape}  (source={source})")

    m = fit_mrd(Y1, Y2, quick=args.quick, messages=args.messages)
    print(f"MRD DeepGP variational bound: {float(m.log_likelihood()):.3f}")
    for i, view in enumerate(m.layers[0].views):
        w = ard_weights(view.kern)
        print(f"View {i} ARD (1/lengthscale): {np.array2string(w, precision=3)}")
    print_ard(m)

    if not args.no_plot or outdir:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        for i, view in enumerate(m.layers[0].views):
            w = ard_weights(view.kern)
            axes[i].bar(np.arange(1, len(w) + 1), w)
            axes[i].set_title(f"view {i} ARD")
            axes[i].set_xlabel("shared dim")
        w_top = ard_weights(m.layers[-1].kern)
        axes[2].bar(np.arange(1, len(w_top) + 1), w_top)
        axes[2].set_title("top layer ARD")
        axes[2].set_xlabel("dimension")
        fig.suptitle("Damianou & Lawrence 2013 — high-five / shared MRD (fig. 4 style)")
        fig.tight_layout()
        save_or_show(fig, outdir, "highfive_ard.png", args.no_plot)

    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(main())
