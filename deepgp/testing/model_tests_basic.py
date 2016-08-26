# Copyright (c) 2015-2016, the authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
import deepgp
import GPy
import os
import h5py
import tables

base_path = os.path.dirname(__file__)

class DGPUnsupervisedTest(unittest.TestCase):
    """
    Normal Deep GPs without back-constraint
    """
    datafile = 'random_10_3.txt'
    modelfile = 'dgp_unsupervised.h5'
    
    def setUp(self):
        # Load data
        Y = np.loadtxt(os.path.join(base_path,self.datafile))
        m = deepgp.DeepGP([Y.shape[1],5,2],Y,kernels=[GPy.kern.RBF(5,ARD=True), GPy.kern.RBF(2,ARD=True)], num_inducing=2, back_constraint=False)
        if not os.path.exists(os.path.join(base_path,self.modelfile)):
            # Create the model file
            m.randomize()
            m._trigger_params_changed()
            m.save(os.path.join(base_path,self.modelfile))
            with h5py.File(os.path.join(base_path,self.modelfile),'r+') as f:
                L = f.create_dataset("L", (1,),dtype=np.float)
                L[:] = m._log_marginal_likelihood
                f.close()
        
        # Load model parameters
        with tables.open_file(os.path.join(base_path,self.modelfile),'r') as f:
            m.param_array[:] = f.root.param_array[:]
            L = float(f.root.L[:])
            m._trigger_params_changed()
            f.close()
        self.model = m
        self.L = L

    def test_gradient(self):
        self.assertTrue(self.model.checkgrad())
        
    def test_checkLogLikelihood(self):
        self.assertTrue(np.allclose(float(self.model._log_marginal_likelihood), self.L, rtol=1e-05, atol=1e-08), str(float(self.model._log_marginal_likelihood))+', '+str(self.L))

class DGPSupervisedTest(unittest.TestCase):
    """
    Normal Deep GPs without back-constraint
    """
    outputfile = 'random_10_3.txt'
    inputfile = 'random_10_2.txt'
    modelfile = 'dgp_supervised.h5'
    
    def setUp(self):
        # Load data
        Y = np.loadtxt(os.path.join(base_path,self.outputfile))
        X = np.loadtxt(os.path.join(base_path,self.inputfile))
        m = deepgp.DeepGP([Y.shape[1],5,X.shape[1]],Y, X=X,kernels=[GPy.kern.RBF(5,ARD=True), GPy.kern.RBF(X.shape[1],ARD=True)], num_inducing=2, back_constraint=False)
        if not os.path.exists(os.path.join(base_path,self.modelfile)):
            # Create the model file
            m.randomize()
            m._trigger_params_changed()
            m.save(os.path.join(base_path,self.modelfile))
            with h5py.File(os.path.join(base_path,self.modelfile),'r+') as f:
                L = f.create_dataset("L", (1,),dtype=np.float)
                L[:] = m._log_marginal_likelihood
                f.close()
                        
        # Load model parameters
        with tables.open_file(os.path.join(base_path,self.modelfile),'r') as f:
            m.param_array[:] = f.root.param_array[:]
            L = float(f.root.L[:])
            m._trigger_params_changed()
            f.close()
        self.model = m
        self.L = L

    def test_gradient(self):
        self.assertTrue(self.model.checkgrad())
        
    def test_checkLogLikelihood(self):
        self.assertTrue(np.allclose(float(self.model._log_marginal_likelihood), self.L, rtol=1e-05, atol=1e-08), str(float(self.model._log_marginal_likelihood))+', '+str(self.L))


class DGP_BC_UnsupervisedTest(unittest.TestCase):
    """
    Normal Deep GPs with back-constraint
    """
    datafile = 'random_10_3.txt'
    modelfile = 'dgp_bc_unsupervised.h5'
    
    def setUp(self):
        # Load data
        Y = np.loadtxt(os.path.join(base_path,self.datafile))
        m = deepgp.DeepGP([Y.shape[1],5,2],Y,kernels=[GPy.kern.RBF(5,ARD=True), GPy.kern.RBF(2,ARD=True)], num_inducing=2, back_constraint=True, encoder_dims=[[3],[2]])
        if not os.path.exists(os.path.join(base_path,self.modelfile)):
            # Create the model file
            m.randomize()
            m._trigger_params_changed()
            m.save(os.path.join(base_path,self.modelfile))
            with h5py.File(os.path.join(base_path,self.modelfile),'r+') as f:
                L = f.create_dataset("L", (1,),dtype=np.float)
                L[:] = m._log_marginal_likelihood
                f.close()
        
        # Load model parameters
        with tables.open_file(os.path.join(base_path,self.modelfile),'r') as f:
            m.param_array[:] = f.root.param_array[:]
            L = float(f.root.L[:])
            m._trigger_params_changed()
            f.close()
        self.model = m
        self.L = L

    def test_gradient(self):
        self.assertTrue(self.model.checkgrad())
        
    def test_checkLogLikelihood(self):
        self.assertTrue(np.allclose(float(self.model._log_marginal_likelihood), self.L, rtol=1e-05, atol=1e-08), str(float(self.model._log_marginal_likelihood))+', '+str(self.L))

class DGP_BC_SupervisedTest(unittest.TestCase):
    """
    Normal Deep GPs with back-constraint
    """
    outputfile = 'random_10_3.txt'
    inputfile = 'random_10_2.txt'
    modelfile = 'dgp_bc_supervised.h5'
    
    def setUp(self):
        # Load data
        Y = np.loadtxt(os.path.join(base_path,self.outputfile))
        X = np.loadtxt(os.path.join(base_path,self.inputfile))
        m = deepgp.DeepGP([Y.shape[1],5,X.shape[1]],Y, X=X,kernels=[GPy.kern.RBF(5,ARD=True), GPy.kern.RBF(X.shape[1],ARD=True)], num_inducing=2, back_constraint=True, encoder_dims=[[3]])
        if not os.path.exists(os.path.join(base_path,self.modelfile)):
            # Create the model file
            m.randomize()
            m._trigger_params_changed()
            m.save(os.path.join(base_path,self.modelfile))
            with h5py.File(os.path.join(base_path,self.modelfile),'r+') as f:
                L = f.create_dataset("L", (1,),dtype=np.float)
                L[:] = m._log_marginal_likelihood
                f.close()
                        
        # Load model parameters
        with tables.open_file(os.path.join(base_path,self.modelfile),'r') as f:
            m.param_array[:] = f.root.param_array[:]
            L = float(f.root.L[:])
            m._trigger_params_changed()
            f.close()
        self.model = m
        self.L = L

    def test_gradient(self):
        self.assertTrue(self.model.checkgrad())
        
    def test_checkLogLikelihood(self):
        self.assertTrue(np.allclose(float(self.model._log_marginal_likelihood), self.L, rtol=1e-05, atol=1e-08), str(float(self.model._log_marginal_likelihood))+', '+str(self.L))


