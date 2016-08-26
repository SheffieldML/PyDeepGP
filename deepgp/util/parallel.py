# Copyright (c) 2015-2016, the authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np

def get_id_within_node(comm=None):
    from mpi4py import MPI
    if comm is None: comm = MPI.COMM_WORLD
    rank = comm.rank
    nodename =  MPI.Get_processor_name()
    nodelist = comm.allgather(nodename)
    return len([i for i in nodelist[:rank] if i==nodename])

def numpy_to_MPI_typemap(np_type):
    from mpi4py import MPI
    typemap = {
        np.dtype(np.float64) : MPI.DOUBLE,
        np.dtype(np.float32) : MPI.FLOAT,
        np.dtype(np.int)     : MPI.INT,
        np.dtype(np.int8)    : MPI.CHAR,
        np.dtype(np.uint8)   : MPI.UNSIGNED_CHAR,
        np.dtype(np.int32)   : MPI.INT,
        np.dtype(np.uint32)  : MPI.UNSIGNED_INT,
    }
    return typemap[np_type]
    

def allReduceArrays(arrL, comm):
    res = []
    for arr in arrL:
        if np.isscalar(arr):
            arr_out = np.empty((1,),dtype=arr.dtype)
            t = numpy_to_MPI_typemap(arr.dtype)
            comm.Allreduce([arr, t], [arr_out, t])
            res.append(arr_out[0])
        else:
            arr_out = arr.copy()
            t = numpy_to_MPI_typemap(arr.dtype)
            comm.Allreduce([arr, t], [arr_out, t])
            res.append(arr_out)
    return res

def reduceArrays(arrL, comm, root=0):
    res = []
    for arr in arrL:
        if np.isscalar(arr):
            arr_out = np.empty((1,),dtype=arr.dtype) if comm.rank==root else None
            t = numpy_to_MPI_typemap(arr.dtype)
            comm.Reduce([arr, t], [arr_out, t], root=root)
            res.append(arr_out[0] if comm.rank==root else None)
        else:
            arr_out = arr.copy() if comm.rank==root else None
            t = numpy_to_MPI_typemap(arr.dtype)
            comm.Reduce([arr, t], [arr_out, t], root=root)
            res.append(arr_out if comm.rank==root else None)
    return res

def broadcastArrays(arrL, comm, root=0):
    for arr in arrL:
        t = numpy_to_MPI_typemap(arr.dtype)
        comm.Bcast(arr, root=root)

