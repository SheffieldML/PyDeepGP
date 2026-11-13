#!/usr/bin/env python
"""Section 4.3 (figs. 6–8): USPS digits {0,1,6} — depth, ARD, 1-NN in latent space.

Paper: Damianou & Lawrence, Deep Gaussian Processes, AISTATS 2013.

Downloads the ESL ``zip.train`` USPS split (16×16), keeps 50 examples of each of
digits {0,1,6}, and trains unsupervised DeepGPs of increasing depth. Reports the
variational bound and leave-one-out 1-NN error in the top latent features.
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
import urllib.request
from typing import List, Sequence, Tuple

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

ESL_ZIP_TRAIN = "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.train.gz"
DIGITS = (0, 1, 6)


def _cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "zip.train.gz")


def download_usps(cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(cache_dir)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    print(f"Downloading USPS zip.train → {path}")
    req = urllib.request.Request(ESL_ZIP_TRAIN, headers={"User-Agent": "PyDeepGP-examples"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(path, "wb") as out:
        out.write(resp.read())
    return path


def load_usps_matrix(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (Y [N,256], labels [N]) with pixels in roughly [-1, 1]."""
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            rows.append([float(x) for x in parts])
    data = np.asarray(rows, dtype=float)
    labels = data[:, 0].astype(int)
    Y = data[:, 1:]
    return Y, labels


def subsample_digits(
    Y: np.ndarray,
    labels: np.ndarray,
    digits: Sequence[int] = DIGITS,
    per_class: int = 50,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    pieces_Y = []
    pieces_L = []
    for d in digits:
        idx = np.where(labels == d)[0]
        if idx.size < per_class:
            raise RuntimeError(f"Digit {d}: need {per_class} examples, found {idx.size}")
        choose = rng.choice(idx, size=per_class, replace=False)
        pieces_Y.append(Y[choose])
        pieces_L.append(np.full(per_class, d, dtype=int))
    return np.vstack(pieces_Y), np.concatenate(pieces_L)


def depth_dimensions(depth: int, data_dim: int, quick: bool) -> List[int]:
    """Latent widths from data up to the top layer (length = depth + 1 entries for DeepGP).

    ``DeepGP(nDims, Y)`` expects ``nDims = [Y_dim, H1, H2, ..., H_depth]``.
    """
    if quick:
        # Narrower / shallower budgets for smoke tests
        widths = {1: [8], 2: [8, 5], 3: [8, 6, 4]}
    else:
        # Overcomplete relative to the true manifold; ARD prunes
        widths = {
            1: [12],
            2: [12, 8],
            3: [12, 10, 6],
            4: [12, 10, 8, 5],
            5: [12, 10, 8, 6, 4],
        }
    if depth not in widths:
        raise ValueError(f"Unsupported depth {depth}; choose from {sorted(widths)}")
    return [data_dim] + widths[depth]


def fit_depth(
    Y: np.ndarray,
    depth: int,
    num_inducing: int,
    iters_fixed: int,
    iters_free: int,
    messages: bool,
    quick: bool,
) -> deepgp.DeepGP:
    dims = depth_dimensions(depth, Y.shape[1], quick=quick)
    kernels = [rbf_bias(q, ard=True) for q in dims[1:]]
    m = deepgp.DeepGP(
        dims,
        Y,
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
        "--depths",
        type=int,
        nargs="+",
        default=None,
        help="Hierarchy depths to fit (default: 1..5, or 1..2 with --quick).",
    )
    p.add_argument("--per-class", type=int, default=50)
    p.add_argument("--num-inducing", type=int, default=40)
    p.add_argument(
        "--cache-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data"),
        help="Directory for zip.train.gz cache.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outdir = ensure_outdir(args.outdir)

    depths = args.depths
    if depths is None:
        depths = [1, 2] if args.quick else [1, 2, 3, 4, 5]

    per_class = 20 if args.quick else args.per_class
    num_inducing = 20 if args.quick else args.num_inducing
    iters_fixed = 40 if args.quick else 200
    iters_free = 100 if args.quick else 2000

    path = download_usps(args.cache_dir)
    Y_all, labels_all = load_usps_matrix(path)
    Y, labels = subsample_digits(Y_all, labels_all, per_class=per_class, seed=args.seed)
    # Zero-mean unit-ish scaling helps optimisation
    Y = Y - Y.mean(0)
    Y = Y / (Y.std() + 1e-12)
    print(f"USPS subset: Y={Y.shape}, classes={sorted(set(labels.tolist()))}")

    results = []
    models = {}
    for depth in depths:
        print(f"\n=== depth {depth} ===")
        m = fit_depth(
            Y,
            depth=depth,
            num_inducing=num_inducing,
            iters_fixed=iters_fixed,
            iters_free=iters_free,
            messages=args.messages,
            quick=args.quick,
        )
        bound = float(m.log_likelihood())
        # Top latent features = last layer's X
        Xtop = np.asarray(m.layers[-1].X.mean)
        mistakes, rate = nearest_neighbour_error(Xtop, labels)
        print(f"bound={bound:.3f}  1-NN mistakes={mistakes}/{Y.shape[0]} ({rate:.3%})")
        print_ard(m)
        results.append((depth, bound, mistakes, rate))
        models[depth] = m

    print("\nSummary")
    print("depth  bound      1-NN errors")
    for depth, bound, mistakes, rate in results:
        print(f"{depth:5d}  {bound:9.3f}  {mistakes:3d} ({rate:.3%})")

    deepest = max(depths)
    m = models[deepest]
    if not args.no_plot or outdir:
        import matplotlib.pyplot as plt

        Xtop = np.asarray(m.layers[-1].X.mean)
        w = ard_weights(m.layers[-1].kern)
        dims = dominant_dims(w, k=2)
        colors = ["C0", "C1", "C2"]
        markers = ["o", "s", "^"]

        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        for i, digit in enumerate(DIGITS):
            mask = labels == digit
            ax.scatter(
                Xtop[mask, dims[0]],
                Xtop[mask, dims[1]],
                c=colors[i],
                marker=markers[i],
                s=28,
                label=str(digit),
            )
        ax.set_xlabel(f"top dim {dims[0]}")
        ax.set_ylabel(f"top dim {dims[1]}")
        ax.set_title(f"USPS top latent (depth {deepest})")
        ax.legend()
        fig.tight_layout()
        save_or_show(fig, outdir, "usps_top_latent.png", args.no_plot)

        fig2, axes = plt.subplots(1, len(m.layers), figsize=(3.2 * len(m.layers), 3))
        if len(m.layers) == 1:
            axes = [axes]
        for i, layer in enumerate(m.layers):
            w = ard_weights(layer.kern)
            axes[i].bar(np.arange(1, len(w) + 1), w)
            axes[i].set_title(f"layer {i} ARD")
            axes[i].set_xlabel("dimension")
        fig2.suptitle("ARD weights (1/lengthscale)")
        fig2.tight_layout()
        save_or_show(fig2, outdir, "usps_ard.png", args.no_plot)

        fig3, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot([r[0] for r in results], [r[1] for r in results], "o-", label="variational bound")
        ax.set_xlabel("depth")
        ax.set_ylabel("bound")
        ax2 = ax.twinx()
        ax2.plot([r[0] for r in results], [r[2] for r in results], "s--", color="C1", label="1-NN mistakes")
        ax2.set_ylabel("1-NN mistakes")
        ax.set_title("Depth selection via bound / 1-NN")
        fig3.tight_layout()
        save_or_show(fig3, outdir, "usps_depth.png", args.no_plot)

    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(main())
