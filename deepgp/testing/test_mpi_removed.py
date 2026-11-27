# Copyright (c) 2015-2016, the authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""CIP-0003 Phase 2: MPI is removed; the arguments remain and fail clearly."""

import unittest
import warnings

import numpy as np
import GPy

import deepgp
from deepgp.util.util import MPI_REMOVED_MESSAGE


def _small_dims_and_data():
    rng = np.random.RandomState(0)
    Y = rng.randn(8, 2)
    X = rng.randn(8, 1)
    kernels = [GPy.kern.RBF(2, ARD=True), GPy.kern.RBF(1, ARD=True)]
    return [2, 2, 1], Y, X, kernels


class MPIRemovedTest(unittest.TestCase):
    def test_mpi_comm_raises_type_error(self):
        nDims, Y, X, kernels = _small_dims_and_data()
        for back_constraint in (True, False):
            with self.assertRaises(TypeError) as ctx:
                deepgp.DeepGP(
                    nDims, Y, X=X, kernels=kernels, num_inducing=3,
                    back_constraint=back_constraint, mpi_comm=object(),
                )
            self.assertIn('psi_chunk_size', str(ctx.exception))
            self.assertEqual(str(ctx.exception), MPI_REMOVED_MESSAGE)

    def test_mpi_comm_none_still_accepted(self):
        nDims, Y, X, kernels = _small_dims_and_data()
        m = deepgp.DeepGP(nDims, Y, X=X, kernels=kernels, num_inducing=3,
                          back_constraint=False, mpi_comm=None, mpi_root=0)
        self.assertTrue(np.isfinite(m.log_likelihood()))

    def test_layer_and_inference_reject_communicator(self):
        from deepgp.inference import VarDTC, SVI_VarDTC, SVI_Ratio
        for cls in (VarDTC, SVI_VarDTC, SVI_Ratio):
            with self.assertRaises(TypeError):
                cls(mpi_comm=object())
        Y = np.random.RandomState(1).randn(6, 2)
        with self.assertRaises(TypeError):
            deepgp.layers.ObservedLayer(2, 1, Y, num_inducing=3,
                                        back_constraint=False, mpi_comm=object())

    def test_modules_removed(self):
        with self.assertRaises(ImportError):
            import deepgp.util.parallel  # noqa: F401
        with self.assertRaises(ImportError):
            from deepgp.inference import VarDTC_parallel  # noqa: F401

    def test_sparse_gp_mpi_alias_warns(self):
        import deepgp.layers.layers as layers
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            cls = layers.SparseGP_MPI
        self.assertIs(cls, layers.SparseGPLayer)
        self.assertTrue(any(issubclass(w.category, FutureWarning) for w in caught))


if __name__ == '__main__':
    unittest.main()
