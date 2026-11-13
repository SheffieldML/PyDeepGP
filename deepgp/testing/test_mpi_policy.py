# Copyright (c) 2015-2016, the authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""CIP-0001 Phase 2: MPI deprecation warning and unsupported configurations."""

import unittest
import warnings

import numpy as np
import GPy

import deepgp
from deepgp.models.model import _MPI_DEPRECATION


def _small_dims_and_data():
    rng = np.random.RandomState(0)
    Y = rng.randn(8, 2)
    X = rng.randn(8, 1)
    kernels = [GPy.kern.RBF(2, ARD=True), GPy.kern.RBF(1, ARD=True)]
    return [2, 2, 1], Y, X, kernels


class MPIPolicyTest(unittest.TestCase):
    def test_mpi_comm_without_back_constraint_raises(self):
        nDims, Y, X, kernels = _small_dims_and_data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            with self.assertRaises(ValueError) as ctx:
                deepgp.DeepGP(
                    nDims, Y, X=X, kernels=kernels, num_inducing=3,
                    back_constraint=False, mpi_comm=object(),
                )
        self.assertIn('back_constraint=False', str(ctx.exception))

    def test_mpi_comm_emits_future_warning(self):
        nDims, Y, X, kernels = _small_dims_and_data()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                deepgp.DeepGP(
                    nDims, Y, X=X, kernels=kernels, num_inducing=3,
                    back_constraint=True, mpi_comm=object(),
                )
            except Exception:
                # Fake communicator is enough to exercise the policy check;
                # construction may fail later when real MPI ops are needed.
                pass
        future = [w for w in caught if issubclass(w.category, FutureWarning)]
        self.assertTrue(future, 'expected FutureWarning for mpi_comm')
        self.assertIn(_MPI_DEPRECATION, str(future[0].message))


if __name__ == '__main__':
    unittest.main()
