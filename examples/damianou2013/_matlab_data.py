"""Loaders and generators matching SheffieldML/deepGP Matlab demos."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def data_path(name: str) -> str:
    return os.path.join(_DATA_DIR, name)


def load_toy_hierarchy(mat_name: str = "hgplvmSampleDataTr1.mat") -> Dict[str, np.ndarray]:
    """Ground-truth hierarchy from ``demToyUnsupervised.m`` / ``hgplvmSampleDataTr1.mat``.

    Returns arrays ``X2`` (top), ``XA``, ``XB`` (intermediate), ``YA``, ``YB`` (leaves).
    """
    path = data_path(mat_name)
    d = sio.loadmat(path, squeeze_me=True)
    out = {k: np.asarray(d[k], dtype=float) for k in ("X2", "XA", "XB", "YA", "YB")}
    for k, arr in out.items():
        if arr.ndim == 1:
            out[k] = arr[:, None]
    return out


def load_highfive(mat_name: str = "demHighFiveHgplvm1.mat") -> Dict[str, np.ndarray]:
    """CMU high-five views from ``demHighFiveDemonstration.m``.

    Returns ``YA``, ``YB`` (78×62) and optional hgplvm latents ``Xparent``,
    ``Xchild1``, ``Xchild2``.
    """
    path = data_path(mat_name)
    d = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    Y = d["Y"]
    out = {
        "YA": np.asarray(Y.YA, dtype=float),
        "YB": np.asarray(Y.YB, dtype=float),
    }
    for name in ("Xparent", "Xchild1", "Xchild2"):
        if hasattr(Y, name):
            out[name] = np.asarray(getattr(Y, name), dtype=float)
    return out


def scale_data(X: np.ndarray) -> np.ndarray:
    """Zero-mean unit-variance columns (Matlab ``scaleData(X,1)``)."""
    X = np.asarray(X, dtype=float)
    X = X - X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return X / std


def generate_hier_gps_new(
    n_toy: int = 150,
    d_toy: int = 15,
    seed: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match ``demToyDynamicsCreateData.m`` branch ``hierGpsNEW``.

    Returns ``(X, Y1_warp, Y)`` where ``X`` is the observed input,
    ``Y1_warp`` is the hidden 1D warp, and ``Y`` is the D-dimensional output.
    """
    import GPy

    rng = np.random.RandomState(seed)
    # Matlab: X = linspace(1,2*pi,Ntoy)'; scaleData
    X = np.linspace(1.0, 2.0 * np.pi, n_toy)[:, None]
    X = scale_data(X)

    # First GP: rbf + lin + white, inverseWidth=4 → lengthscale = 0.5
    kern1 = (
        GPy.kern.RBF(1, variance=1.0, lengthscale=0.5)
        + GPy.kern.Linear(1, variances=1.0)
        + GPy.kern.White(1, variance=1e-6)
    )
    K1 = kern1.K(X)
    # gsamp: one sample from N(0,K)
    Y1 = rng.multivariate_normal(np.zeros(n_toy), K1)[:, None]

    # Second GP: rbf on Y1, inverseWidth=16 → lengthscale=0.25
    kern2 = GPy.kern.RBF(1, variance=1.0, lengthscale=0.25) + GPy.kern.White(
        1, variance=1e-6
    )
    K2 = kern2.K(Y1)
    # D independent samples
    L2 = np.linalg.cholesky(K2 + 1e-10 * np.eye(n_toy))
    Y = L2 @ rng.randn(n_toy, d_toy)
    return X, Y1, Y


def train_test_split_indices(n_toy: int, n_tr: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Match ``demToyDynamicsSplitDataset.m`` (non-block): random ``Ntr`` train indices."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_toy)
    ind_tr = np.sort(perm[:n_tr])
    ind_ts = np.sort(perm[n_tr:])
    return ind_tr, ind_ts


def load_usps_vercat2(
    per_class: int = 50,
    cache_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """USPS digits {0,1,6} as in ``demUsps.m`` / ``demDigitsDemonstration.m`` ``vercat2``.

    Matlab indexing (1-based contiguous blocks in the ESL / GPmat layout) is
    approximated by taking the first ``per_class`` examples of each digit from
    ESL ``zip.train`` (same counts; order may differ slightly from GPmat).
    """
    import gzip
    import urllib.request

    if cache_dir is None:
        cache_dir = _DATA_DIR
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "zip.train.gz")
    if not (os.path.isfile(path) and os.path.getsize(path) > 0):
        url = "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.train.gz"
        req = urllib.request.Request(url, headers={"User-Agent": "PyDeepGP-damianou2013"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(path, "wb") as out:
            out.write(resp.read())

    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                rows.append([float(x) for x in parts])
    data = np.asarray(rows, dtype=float)
    labels = data[:, 0].astype(int)
    Y = data[:, 1:]

    pieces_Y: List[np.ndarray] = []
    pieces_L: List[np.ndarray] = []
    # Matlab vercat2 order: 0, 6, 1
    for digit in (0, 6, 1):
        idx = np.where(labels == digit)[0]
        if idx.size < per_class:
            raise RuntimeError(f"Need {per_class} examples of digit {digit}, found {idx.size}")
        # Take the first per_class in file order (stable, closer to block loads)
        choose = idx[:per_class]
        pieces_Y.append(Y[choose])
        pieces_L.append(np.full(per_class, digit, dtype=int))
    return np.vstack(pieces_Y), np.concatenate(pieces_L)


# Matlab demDigitsDemonstration experiment 19 widths
USPS_DEPTH_Q = {
    1: [15],
    2: [15, 12],
    3: [15, 12, 10],
    4: [15, 12, 10, 8],
    5: [15, 12, 10, 8, 6],
}
