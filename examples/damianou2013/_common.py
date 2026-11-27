"""Shared helpers for Damianou & Lawrence (AISTATS 2013) recreation scripts."""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import numpy as np

import GPy
import deepgp


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fewer iterations / smaller models for a smoke test (not paper-faithful).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed (default: 42).",
    )
    parser.add_argument(
        "--messages",
        action="store_true",
        help="Show optimiser progress messages.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip matplotlib figures.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="If set, save figures under this directory.",
    )
    return parser


def ensure_outdir(outdir: Optional[str]) -> Optional[str]:
    if outdir is None:
        return None
    os.makedirs(outdir, exist_ok=True)
    return outdir


def save_or_show(fig, outdir: Optional[str], name: str, no_plot: bool) -> None:
    import matplotlib.pyplot as plt

    if outdir is not None:
        path = os.path.join(outdir, name)
        fig.savefig(path, bbox_inches="tight", dpi=120)
        print(f"Wrote {path}")
    if no_plot:
        plt.close(fig)
    else:
        plt.show()


def rbf_bias(input_dim: int, ard: bool = True) -> GPy.kern.Kern:
    return GPy.kern.RBF(input_dim, ARD=ard) + GPy.kern.Bias(input_dim)


def _output_variance(y) -> float:
    """Variance of observed data or of a variational posterior mean."""
    from GPy.core.parameterization.variational import NormalPosterior, VariationalPosterior

    if isinstance(y, (NormalPosterior, VariationalPosterior)):
        return float(np.asarray(y.mean).var())
    return float(np.asarray(y).var())


def init_noise(model: deepgp.DeepGP, fraction: float = 0.01, fix: bool = False) -> None:
    for layer in model.layers:
        if hasattr(layer, "views"):
            for view in layer.views:
                view.Gaussian_noise.variance = _output_variance(view.Y) * fraction
                if fix:
                    view.Gaussian_noise.variance.fix()
        else:
            layer.Gaussian_noise.variance = _output_variance(layer.Y) * fraction
            if fix:
                layer.Gaussian_noise.variance.fix()


def unfix_noise(model: deepgp.DeepGP) -> None:
    for layer in model.layers:
        if hasattr(layer, "views"):
            for view in layer.views:
                view.Gaussian_noise.variance.unfix()
        else:
            layer.Gaussian_noise.variance.unfix()


def optimize_staged(
    model: deepgp.DeepGP,
    iters_fixed: int,
    iters_free: int,
    messages: bool = False,
) -> None:
    """Fix noise briefly, then free it — matches the shipped supervised examples."""
    init_noise(model, fix=True)
    if iters_fixed > 0:
        model.optimize(max_iters=iters_fixed, messages=messages)
    unfix_noise(model)
    if iters_free > 0:
        model.optimize(max_iters=iters_free, messages=messages)


def lengthscales(kern: GPy.kern.Kern) -> np.ndarray:
    """Return ARD lengthscales from an RBF(+Bias) kernel."""
    if hasattr(kern, "rbf"):
        return np.asarray(kern.rbf.lengthscale).ravel()
    if hasattr(kern, "lengthscale"):
        return np.asarray(kern.lengthscale).ravel()
    # Additive kernels: prefer the first RBF-like part
    if hasattr(kern, "parts"):
        for part in kern.parts:
            if hasattr(part, "lengthscale"):
                return np.asarray(part.lengthscale).ravel()
    raise AttributeError("Could not find lengthscale on kernel")


def ard_weights(kern: GPy.kern.Kern) -> np.ndarray:
    """Inverse lengthscale (larger ⇒ more relevant), as used in the paper plots."""
    ls = lengthscales(kern)
    return 1.0 / np.maximum(ls, 1e-12)


def print_ard(model: deepgp.DeepGP) -> None:
    for i, layer in enumerate(model.layers):
        try:
            w = ard_weights(layer.kern)
        except AttributeError:
            continue
        print(f"Layer {i} ARD (1/lengthscale): {np.array2string(w, precision=3)}")


def nearest_neighbour_error(X: np.ndarray, labels: np.ndarray) -> Tuple[int, float]:
    """Leave-one-out 1-NN error count and rate in Euclidean feature space."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels).ravel()
    n = X.shape[0]
    # Squared distances via (x-y)^2 = x^2 + y^2 - 2xy
    sq = np.sum(X * X, axis=1, keepdims=True)
    d2 = sq + sq.T - 2.0 * X @ X.T
    np.fill_diagonal(d2, np.inf)
    nn = np.argmin(d2, axis=1)
    mistakes = int(np.sum(labels[nn] != labels))
    return mistakes, mistakes / float(n)


def dominant_dims(weights: np.ndarray, k: int = 2) -> List[int]:
    order = np.argsort(-weights)
    return [int(i) for i in order[:k]]


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((np.asarray(pred) - np.asarray(target)) ** 2))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(mse(pred, target)))


def sample_gp_prior(
    X: np.ndarray,
    kernel: GPy.kern.Kern,
    output_dim: int,
    noise: float = 1e-6,
) -> np.ndarray:
    """Draw ``output_dim`` independent prior samples at inputs ``X``."""
    K = kernel.K(X)
    K = K + noise * np.eye(X.shape[0])
    L = np.linalg.cholesky(K)
    Z = np.random.randn(X.shape[0], output_dim)
    return L @ Z
