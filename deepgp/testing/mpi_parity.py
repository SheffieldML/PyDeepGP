# Copyright (c) 2015-2016, the authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
CIP-0001 R11: compare 2-rank MPI objective with a serial back-constrained model.

Run under MPI::

    mpiexec -n 2 python -m deepgp.testing.mpi_parity

Exits 0 when the MPI log-likelihood on root matches the serial reference
within a relative tolerance; otherwise exits 1.
"""

from __future__ import print_function

import sys
import warnings

import numpy as np
import GPy

import deepgp


def _build_model(Y, X, mpi_comm=None):
    kernels = [GPy.kern.RBF(3, ARD=True), GPy.kern.RBF(2, ARD=True)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        return deepgp.DeepGP(
            [2, 3, 2],
            Y,
            X=X,
            kernels=kernels,
            num_inducing=4,
            back_constraint=True,
            mpi_comm=mpi_comm,
            mpi_root=0,
        )


def main():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    if size < 2:
        if rank == 0:
            print('mpi_parity requires at least 2 ranks', file=sys.stderr)
        return 2

    rng = np.random.RandomState(42)
    N, D, Q = 20, 2, 2
    Y_all = rng.randn(N, D)
    X_all = rng.randn(N, Q)

    # Contiguous row partition across ranks (same total data as serial).
    counts = [N // size + (1 if r < N % size else 0) for r in range(size)]
    starts = np.cumsum([0] + counts[:-1])
    sl = slice(int(starts[rank]), int(starts[rank] + counts[rank]))
    Y_local = np.ascontiguousarray(Y_all[sl])
    X_local = np.ascontiguousarray(X_all[sl])

    serial_ll = None
    if rank == 0:
        m_serial = _build_model(Y_all, X_all, mpi_comm=None)
        serial_ll = float(m_serial.log_likelihood())
    serial_ll = comm.bcast(serial_ll, root=0)

    m_mpi = _build_model(Y_local, X_local, mpi_comm=comm)
    # After internal Allreduce of psi statistics, root holds the global objective.
    mpi_obj = float(m_mpi.log_likelihood()) if rank == 0 else None
    if rank == 0:
        rel = abs(mpi_obj - serial_ll) / max(1.0, abs(serial_ll))
        print('serial_ll={:.10g} mpi_ll={:.10g} rel_diff={:.3e}'.format(
            serial_ll, mpi_obj, rel))
        if not np.isfinite(mpi_obj) or rel > 1e-5:
            print('MPI parity check failed', file=sys.stderr)
            return 1
        print('MPI parity check passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
