# Copyright (c) 2015-2016, the authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""Helpers for CIP-0001 Phase 3 chunked psi-statistics."""

from __future__ import print_function


def row_slices(N, chunk_size):
    """Yield contiguous row slices covering ``0..N-1``."""
    for start in range(0, N, int(chunk_size)):
        yield slice(start, min(start + int(chunk_size), N))


def configure_kern_psicomp(kern, chunk_size=None, n_threads=1):
    """Apply chunking knobs to a kernel's ``psicomp`` if supported (GPy R12).

    Returns True when the kernel will honour ``chunk_size`` / ``n_threads``.
    """
    psicomp = getattr(kern, "psicomp", None)
    if psicomp is None or not hasattr(psicomp, "chunk_size"):
        return False
    psicomp.chunk_size = chunk_size
    psicomp.n_threads = 1 if n_threads is None else int(n_threads)
    return True


def apply_psi_chunking(layer, chunk_size=None, n_threads=1):
    """Configure a layer's inference method and kernel for psi chunking."""
    inf = getattr(layer, "inference_method", None)
    if inf is not None:
        inf.psi_chunk_size = chunk_size
        inf.psi_threads = 1 if n_threads is None else int(n_threads)
    kern = getattr(layer, "kern", None)
    if kern is not None:
        configure_kern_psicomp(kern, chunk_size, n_threads)
