# Copyright (c) 2015-2016, the authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np


def get_id_within_node(comm=None):
    from mpi4py import MPI
    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.rank
    nodename = MPI.Get_processor_name()
    nodelist = comm.allgather(nodename)
    return len([i for i in nodelist[:rank] if i == nodename])


def allReduceArrays(arrL, comm):
    """Allreduce a list of NumPy scalars/arrays. mpi4py infers MPI datatypes."""
    res = []
    for arr in arrL:
        if np.isscalar(arr):
            send = np.asarray(arr).reshape(1)
            arr_out = np.empty_like(send)
            comm.Allreduce(send, arr_out)
            res.append(arr_out.reshape(()))
        else:
            arr_out = np.empty_like(arr)
            comm.Allreduce(arr, arr_out)
            res.append(arr_out)
    return res


def reduceArrays(arrL, comm, root=0):
    """Reduce a list of NumPy scalars/arrays to ``root``. Non-root entries are None."""
    res = []
    for arr in arrL:
        if np.isscalar(arr):
            send = np.asarray(arr).reshape(1)
            arr_out = np.empty_like(send) if comm.rank == root else None
            comm.Reduce(send, arr_out, root=root)
            res.append(arr_out.reshape(()) if comm.rank == root else None)
        else:
            arr_out = np.empty_like(arr) if comm.rank == root else None
            comm.Reduce(arr, arr_out, root=root)
            res.append(arr_out if comm.rank == root else None)
    return res


def broadcastArrays(arrL, comm, root=0):
    """Broadcast a list of NumPy arrays from ``root`` (in-place)."""
    for arr in arrL:
        comm.Bcast(arr, root=root)
