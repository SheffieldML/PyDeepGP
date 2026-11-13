#!/usr/bin/env python
"""Section 4.2 (figs. 4–5): CMU high-five — shared / MRD hierarchy.

Paper: Damianou & Lawrence, Deep Gaussian Processes, AISTATS 2013.

The paper models two interacting subjects as conditionally independent views
that share a parent latent layer (an MRD-style horizontal split of the
observation layer). ``DeepGP`` still accepts ``Y`` as a list and builds an
``ObservedMRDLayer``, but that path is outside the maintained/tested surface
(see project README / CIP-0001 coverage notes).

This script:

* documents the intended experiment,
* optionally loads two-view MOCAP via ``pods`` / GPy utilities when available,
* exits with a clear status unless ``--force-mrd`` is passed.

Use ``--force-mrd`` only for exploratory revival work; results are not covered
by CI.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap


MESSAGE = textwrap.dedent(
    """
    High-five / shared-hierarchy experiment (Damianou & Lawrence 2013, §4.2)
    -----------------------------------------------------------------------
    Status: not recreated on the maintained DeepGP path.

    Required capability: conditionally independent observation views with a
    shared parent latent layer (MRD-style). Code exists in
    ``deepgp.layers.mrd.ObservedMRDLayer`` and is reachable via
    ``DeepGP([ [D1, D2], Q1, Q2 ], Y=[Y1, Y2], ...)``, but MRD is omitted from
    unit tests and coverage.

    To revive this experiment:
      1. Re-establish MRD tests and a documented construction API.
      2. Load CMU MOCAP high-five (two subjects, ~78 frames × 62 dims each;
         Lawrence & Moore 2007 / paper §4.2).
      3. Fit with separate ARD kernels per view; inspect shared dimensions.

    Related single-view demos that *do* run today:
      examples/example_unsupervised_learning.py
      examples/example_supervised_learning.py  (walk/jog feature split)
    """
).strip()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--force-mrd",
        action="store_true",
        help="Attempt an experimental two-view DeepGP fit (unmaintained).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--messages", action="store_true")
    p.add_argument("--quick", action="store_true")
    return p


def _try_load_two_views(seed: int):
    """Best-effort loader; returns (Y1, Y2) or raises."""
    import numpy as np

    try:
        import pods

        # Prefer a CMU sequence if pods exposes one; otherwise synthesise two
        # correlated views so --force-mrd can exercise the API.
        if hasattr(pods.datasets, "cmu_mocap_35_walk_jog"):
            data = pods.datasets.cmu_mocap_35_walk_jog()
            Y = np.asarray(data["Y"], dtype=float)
            n = min(80, Y.shape[0])
            Y = Y[:n]
            mid = Y.shape[1] // 2
            return Y[:, :mid], Y[:, mid:]
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"pods MOCAP unavailable ({exc}); using synthetic correlated views.")

    rng = np.random.RandomState(seed)
    n, q, d = 60, 3, 20
    X = rng.randn(n, q)
    W1, W2 = rng.randn(q, d), rng.randn(q, d)
    Y1 = X @ W1 + 0.05 * rng.randn(n, d)
    Y2 = X @ W2 + 0.05 * rng.randn(n, d)
    return Y1, Y2


def run_force_mrd(quick: bool, seed: int, messages: bool) -> int:
    import deepgp

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _common import init_noise, rbf_bias, unfix_noise

    Y1, Y2 = _try_load_two_views(seed)
    Y1 = (Y1 - Y1.mean(0)) / (Y1.std() + 1e-12)
    Y2 = (Y2 - Y2.mean(0)) / (Y2.std() + 1e-12)
    q1, q2 = (4, 3) if quick else (6, 4)
    num_inducing = 15 if quick else 30
    print(f"Experimental MRD DeepGP on views {Y1.shape} and {Y2.shape}")

    # ObservedMRDLayer requires one kernel per view (input dim = shared parent width).
    kernels = [
        [rbf_bias(q1, ard=True), rbf_bias(q1, ard=True)],
        rbf_bias(q2, ard=True),
    ]
    try:
        m = deepgp.DeepGP(
            [[Y1.shape[1], Y2.shape[1]], q1, q2],
            [Y1, Y2],
            kernels=kernels,
            num_inducing=num_inducing,
            back_constraint=False,
        )
        init_noise(m, fix=True)
        m.optimize(max_iters=40 if quick else 200, messages=messages)
        unfix_noise(m)
        m.optimize(max_iters=80 if quick else 800, messages=messages)
    except Exception as exc:
        print(
            "MRD construction/optimisation failed (expected while MRD is unmaintained):\n"
            f"  {type(exc).__name__}: {exc}"
        )
        return 1
    print(f"MRD DeepGP variational bound: {float(m.log_likelihood()):.3f}")
    print("Note: this path is unmaintained; treat results as exploratory only.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(MESSAGE)
    if not args.force_mrd:
        print("\nRe-run with --force-mrd to attempt the experimental MRD fit.")
        return 2
    return run_force_mrd(quick=args.quick, seed=args.seed, messages=args.messages)


if __name__ == "__main__":
    raise SystemExit(main())
