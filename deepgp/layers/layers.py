# Copyright (c) 2015-2016, the authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np

from GPy.core import SparseGP, Model, Param
from GPy.core.parameterization.transformations import Logexp
from paramz.transformations import __fixed__
from GPy import likelihoods
from GPy import kern
from GPy.core.parameterization.variational import NormalPosterior, VariationalPosterior
from deepgp.util.variational import NormalEntropy,NormalPrior
from deepgp.util.parallel import reduceArrays
from GPy.inference.latent_function_inference.posterior import Posterior

class SparseGP_MPI(SparseGP):
    
    def __init__(self, X, Y, Z, kernel, likelihood,
                 mean_function=None, inference_method=None,
                 name='sparse gp', Y_metadata=None,
                 normalizer=False,
                 mpi_comm=None,
                 mpi_root=0,
                 auto_update=True):
        self.mpi_comm = mpi_comm
        self.mpi_root = mpi_root
        self.psicov = False
        self.svi = False
        self.qU_ratio = 1.
        self.auto_update = auto_update

        if inference_method is None:
            from ..inference import VarDTC_parallel, VarDTC
            if mpi_comm is None:
                inference_method = VarDTC()
            else:
                inference_method = VarDTC_parallel(mpi_comm, mpi_root)
        elif inference_method=='inferentia' and mpi_comm is None:
            from ..inference import VarDTC_Inferentia
            inference_method = VarDTC_Inferentia()
            self.psicov = True
        elif inference_method=='svi':
            from ..inference import SVI_VarDTC
            inference_method = SVI_VarDTC()
            self.svi = True
        
        super(SparseGP_MPI, self).__init__(X, Y, Z, kernel,
                                           likelihood,
                                           mean_function=mean_function,
                                           inference_method=inference_method,
                                           name=name, Y_metadata=Y_metadata,
                                           normalizer=normalizer)
        
        if self.svi:
            from ..util.misc import comp_mapping
            W = comp_mapping(self.X, self.Y)
            qu_mean = self.Z.dot(W)
            self.qU_mean = Param('qU_m', qu_mean)
            self.qU_W = Param('qU_W', np.random.randn(Z.shape[0], Z.shape[0])*0.01) 
            self.qU_a = Param('qU_a', 1e-3, Logexp())
            self.link_parameters(self.qU_mean, self.qU_W, self.qU_a)

    def parameters_changed(self):
        if self.auto_update: self.update_layer()
        
    def update_layer(self):
        self._inference_vardtc()

    def _inference_vardtc(self):
        if self.svi:
            from GPy.util.linalg import tdot
            self.qU_var = tdot(self.qU_W)+np.eye(self.Z.shape[0])*self.qU_a
            self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(self.kern, self.X, self.Z, self.likelihood, self.Y, self.qU_mean , self.qU_var, Kuu_sigma=self.Kuu_sigma)
            
            if self.mpi_comm is None or (self.mpi_comm is not None and self.mpi_comm.rank==self.mpi_root):
                KL, dKL_dqU_mean, dKL_dqU_var, dKL_dKuu = self.inference_method.comp_KL_qU(self.qU_mean ,self.qU_var)
                self._log_marginal_likelihood += -KL*self.qU_ratio        
                self.grad_dict['dL_dqU_mean'] += -dKL_dqU_mean*self.qU_ratio
                self.grad_dict['dL_dqU_var'] += -dKL_dqU_var*self.qU_ratio
                self.grad_dict['dL_dKmm'] += -dKL_dKuu*self.qU_ratio
        else:
            self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(self.kern, self.X, self.Z, self.likelihood, self.Y, self.Y_metadata, Kuu_sigma=self.Kuu_sigma if hasattr(self, 'Kuu_sigma') else None)

        self.likelihood.update_gradients(self.grad_dict['dL_dthetaL'])
        dL_dKmm = self.grad_dict['dL_dKmm']
        if (self.mpi_comm is None or (self.mpi_comm is not None and self.mpi_comm.rank==self.mpi_root)) and (hasattr(self, 'Kuu_sigma') and self.Kuu_sigma is not None):
            self.Kuu_sigma.gradient = np.diag(dL_dKmm)

        if isinstance(self.X, VariationalPosterior):
            #gradients wrt kernel
            
            if self.psicov:
                self.kern.update_gradients_expectations_psicov(variational_posterior=self.X,
                                                        Z=self.Z,
                                                        dL_dpsi0=self.grad_dict['dL_dpsi0'],
                                                        dL_dpsi1=self.grad_dict['dL_dpsi1'],
                                                        dL_dpsicov=self.grad_dict['dL_dpsicov'])
            else:
                self.kern.update_gradients_expectations(variational_posterior=self.X,
                                                        Z=self.Z,
                                                        dL_dpsi0=self.grad_dict['dL_dpsi0'],
                                                        dL_dpsi1=self.grad_dict['dL_dpsi1'],
                                                        dL_dpsi2=self.grad_dict['dL_dpsi2'])
            kerngrad = self.kern.gradient.copy()
            if self.mpi_comm is None:
                self.kern.update_gradients_full(dL_dKmm, self.Z, None)
                kerngrad += self.kern.gradient.copy()
                self.kern.gradient = kerngrad
            else:
                kerngrad = reduceArrays([kerngrad], self.mpi_comm, self.mpi_root)[0]
                if self.mpi_comm.rank==self.mpi_root:
                    self.kern.update_gradients_full(dL_dKmm, self.Z, None)
                    kerngrad += self.kern.gradient.copy()
                    self.kern.gradient = kerngrad

            #gradients wrt Z
            if self.psicov:
                self.Z.gradient = self.kern.gradients_Z_expectations_psicov(
                                   self.grad_dict['dL_dpsi0'],
                                   self.grad_dict['dL_dpsi1'],
                                   self.grad_dict['dL_dpsicov'],
                                   Z=self.Z,
                                   variational_posterior=self.X)
            else:
                self.Z.gradient = self.kern.gradients_Z_expectations(
                                   self.grad_dict['dL_dpsi0'],
                                   self.grad_dict['dL_dpsi1'],
                                   self.grad_dict['dL_dpsi2'],
                                   Z=self.Z,
                                   variational_posterior=self.X)
            if self.mpi_comm is None:
                self.Z.gradient += self.kern.gradients_X(dL_dKmm, self.Z)
            else:
                self.Z.gradient =  reduceArrays([self.Z.gradient], self.mpi_comm, self.mpi_root)[0]
                if self.mpi_comm.rank == self.mpi_root:
                    self.Z.gradient += self.kern.gradients_X(dL_dKmm, self.Z)
        else:
            #gradients wrt kernel
            self.kern.update_gradients_diag(self.grad_dict['dL_dKdiag'], self.X)
            kerngrad = self.kern.gradient.copy()
            self.kern.update_gradients_full(self.grad_dict['dL_dKnm'], self.X, self.Z)
            kerngrad += self.kern.gradient
            if self.mpi_comm is None:
                self.kern.update_gradients_full(dL_dKmm, self.Z, None)
                self.kern.gradient += kerngrad
            else:
                kerngrad = reduceArrays([kerngrad], self.mpi_comm, self.mpi_root)[0]
                if self.mpi_comm.rank==self.mpi_root:
                    self.kern.update_gradients_full(dL_dKmm, self.Z, None)
                    kerngrad += self.kern.gradient.copy()
                    self.kern.gradient = kerngrad
            #gradients wrt Z
            self.Z.gradient = self.kern.gradients_X(self.grad_dict['dL_dKnm'].T, self.Z, self.X)
            if self.mpi_comm is None:
                self.Z.gradient += self.kern.gradients_X(dL_dKmm, self.Z)
            else:
                self.Z.gradient =  reduceArrays([self.Z.gradient], self.mpi_comm, self.mpi_root)[0]
                if self.mpi_comm.rank == self.mpi_root:
                    self.Z.gradient += self.kern.gradients_X(dL_dKmm, self.Z)

        if self.svi:
            self.qU_mean.gradient = self.grad_dict['dL_dqU_mean']
            self.qU_W.gradient = (self.grad_dict['dL_dqU_var']+self.grad_dict['dL_dqU_var'].T).dot(self.qU_W)
            self.qU_a.gradient = np.diag(self.grad_dict['dL_dqU_var']).sum()
            
class Layer(SparseGP_MPI):
    
    def __init__(self, layer_lower,
                 dim_down, dim_up,
                 likelihood,
                 X=None, X_variance=None, init='PCA',
                 Z=None, num_inducing=10,  kernel=None,
                 inference_method=None, uncertain_inputs=True,
                 mpi_comm=None, mpi_root=0, back_constraint=True,
                 encoder=None, auto_update=True, name='layer'):

        self.uncertain_inputs = uncertain_inputs
        self.layer_lower = layer_lower
        Y = self.Y if self.layer_lower is None else self.layer_lower.X
        self.back_constraint = back_constraint

        from deepgp.util.util import initialize_latent
        if X is None: X, _ = initialize_latent(init, Y.shape[0], dim_up, Y.mean.values if isinstance(Y, VariationalPosterior) else Y)
        if X_variance is None: X_variance = 0.01*np.ones(X.shape) + 0.01*np.random.rand(*X.shape)
            
        if Z is None:
            if self.back_constraint: Z = np.random.rand(num_inducing, dim_up)*2-1.
            else:
                if num_inducing<=X.shape[0]:
                    Z = X[np.random.permutation(X.shape[0])[:num_inducing]].copy()
                else:
                    Z_more = np.random.rand(num_inducing-X.shape[0],X.shape[1])*(X.max(0)-X.min(0))+X.min(0)
                    Z = np.vstack([X.copy(),Z_more])
        assert Z.shape[1] == X.shape[1]
        
        if mpi_comm is not None:
            from ..util.parallel import broadcastArrays
            broadcastArrays([Z], mpi_comm, mpi_root)
        
        if uncertain_inputs: X = NormalPosterior(X, X_variance)
        if kernel is None: kernel = kern.RBF(dim_up, ARD = True)
        assert kernel.input_dim==X.shape[1], "The dimensionality of input has to be equal to the input dimensionality of the kernel!"
        self.Kuu_sigma = Param('Kuu_var', np.zeros(num_inducing)+1e-3, Logexp())
        
        super(Layer, self).__init__(X, Y, Z, kernel,
                                    likelihood, inference_method=inference_method,
                                    mpi_comm=mpi_comm, mpi_root=mpi_root,
                                    auto_update=auto_update, name=name)
        self.link_parameter(self.Kuu_sigma)
        if back_constraint: self.encoder = encoder

        if self.uncertain_inputs and not self.back_constraint:
            self.link_parameter(self.X)

    @property
    def Y(self):
        if self.layer_lower is None:
            return self._Y
        else:
            if hasattr(self.layer_lower,'repeatX') and self.layer_lower.repeatX:
                return self.layer_lower.X[:,:self.layer_lower.repeatXsplit]
            else:
                return self.layer_lower.X
    
    @Y.setter
    def Y(self, value):
        if self.layer_lower is None:
            self._Y = value
        # else:
        #     if hasattr(self.layer_lower,'repeatX') and self.layer_lower.repeatX:
        #         self.layer_lower.X[:self.layer_lower.repeatXsplit:] = value
        #     else:
        #         self.layer_lower.X = value
    
    # Wrapper function which returns single points of the observation space of this layer.
    @property
    def Y_vals(self):
        # Perhaps we shouldn't make this a function
        return self.Y if self.layer_lower is None else self.layer_lower.X.mean   
        
    def update_qX_gradients(self):
        if self.psicov:
            self.X.mean.gradient, self.X.variance.gradient = self.kern.gradients_qX_expectations_psicov(
                                                variational_posterior=self.X,
                                                Z=self.Z,
                                                dL_dpsi0=self.grad_dict['dL_dpsi0'],
                                                dL_dpsi1=self.grad_dict['dL_dpsi1'],
                                                dL_dpsicov=self.grad_dict['dL_dpsicov'])
        else:
            self.X.mean.gradient, self.X.variance.gradient = self.kern.gradients_qX_expectations(
                                                variational_posterior=self.X,
                                                Z=self.Z,
                                                dL_dpsi0=self.grad_dict['dL_dpsi0'],
                                                dL_dpsi1=self.grad_dict['dL_dpsi1'],
                                                dL_dpsi2=self.grad_dict['dL_dpsi2'])
        delta = -self.variationalterm.comp_value(self.X)
        if self.mpi_comm is not None:
            delta = reduceArrays([np.float64(delta)],self.mpi_comm, self.mpi_root)[0]
            if self.mpi_comm.rank != self.mpi_root: delta = 0.
        self._log_marginal_likelihood += delta
        self.variationalterm.update_gradients(self.X)
            
    def update_layer(self):
        super(Layer,self).update_layer()
        if self.uncertain_inputs:
            self.update_qX_gradients()
            
    def gen_pred_layer(self, layer_lower=None, Y=None, X=None, binObserved=False):
        from .pred_layers import PredLayer, BinaryPredLayer
        from deepgp.encoder.mlp import MLP
        from ..inference import SVI_Ratio, SVI_Ratio_Binary
        from copy import deepcopy
        
        X = self.X.copy() if X is None else X
        Y = self.Y.copy() if Y is None else Y
        Z = self.Z.values.copy()
        X_var = self.X_var.values.copy() if self.back_constraint else None
        encoder = MLP.clone(self.encoder) if self.back_constraint else None
        kernel = self.kern.copy()
        likelihood = self.likelihood.copy()
        posterior = deepcopy(self.posterior)
        variationalterm = NormalPrior() if isinstance(self.variationalterm, NormalPrior) else NormalEntropy()
        
        if binObserved:
            layer = BinaryPredLayer(X, Y, kernel, Z,
                                    posterior,
                                    likelihood=likelihood,
                                    layer_lower=layer_lower,
                                    inference_method=SVI_Ratio_Binary(), 
                                    variationalterm=variationalterm,
                                    X_var=X_var,
                                    encoder=encoder,
                                    name=self.name)
        else:
            layer = PredLayer(X, Y, kernel, Z, posterior,
                              likelihood=likelihood,
                              layer_lower=layer_lower,
                              inference_method=SVI_Ratio(), 
                              variationalterm=variationalterm,
                              X_var=X_var,
                              encoder=encoder, name=self.name)
        return layer
    
    def set_newX(self, X, append=False):
        from GPy import ObsAr
        if not self.uncertain_inputs:
            if append:
                self.X = ObsAr(np.vstack([self.X, X]))
            else:
                self.X = X if isinstance(X,ObsAr) else ObsAr(X)
        else:
            self.unlink_parameter(self.X)
            if append:
                self.X = NormalPosterior(np.vstack([self.X.mean.values, X.mean.values]),
                                         np.vstack([self.X.variance.values, X.variance.values]))
            else:
                self.X = X
            self.link_parameter(self.X)

class ObservedLayer(Layer):

    def __init__(self, dim_down, dim_up,
                 Y, X=None, X_variance=None,
                 Z=None, num_inducing=10,
                 kernel=None, inference_method=None,
                 likelihood=None, init='rand', 
                    mpi_comm=None, mpi_root=0,
                 back_constraint=True, encoder=None,
                 auto_update=True, repeatX=False,
                 repeatXsplit=0, name='obslayer'):
        self.dim_up, self.dim_down = dim_up, dim_down
        self._Y = Y
        self.repeatX = repeatX
        self.repeatXsplit = repeatXsplit
        if likelihood is None:  likelihood = likelihoods.Gaussian()
        self._toplayer_ = False
        self.variationalterm = NormalEntropy()
        super(ObservedLayer, self).__init__(None, self.dim_down, dim_up,
                                            likelihood, init=init, X=X,
                                            X_variance=X_variance, Z=Z, 
                                            num_inducing=num_inducing,
                                            kernel=kernel,
                                            inference_method=inference_method,
                                            mpi_comm=mpi_comm, mpi_root=mpi_root,
                                            back_constraint=back_constraint,
                                            encoder=encoder, auto_update=auto_update,
                                            name=name)
        
    def set_as_toplayer(self, flag=True):
        if flag:
            self.variationalterm = NormalPrior()
        else:
            self.variationalterm = NormalEntropy()
        self._toplayer_ = flag


class HiddenLayer(Layer):

    def __init__(self, layer_lower, dim_up,
                 X=None, X_variance=None,
                 Z=None, num_inducing=10,
                 kernel=None, inference_method=None,
                 noise_var=1e-2, init='rand',
                 mpi_comm=None, mpi_root=0, back_constraint=True,
                 encoder=None, auto_update=True, name='hiddenlayer'):
        
        self.dim_up, self.dim_down = dim_up, layer_lower.X.shape[1] #self.Y.shape[1]
        likelihood = likelihoods.Gaussian(variance=noise_var)
        self.variationalterm = NormalEntropy()

        super(HiddenLayer, self).__init__(layer_lower, self.dim_down,
                                          dim_up, likelihood, init=init,
                                          X=X, X_variance=X_variance, Z=Z,
                                          num_inducing=num_inducing,
                                          kernel=kernel,
                                          inference_method=inference_method,
                                          mpi_comm=mpi_comm, mpi_root=mpi_root,
                                          back_constraint=back_constraint,
                                          encoder=encoder, auto_update=auto_update,
                                          name=name)

    def update_layer(self):
        super(HiddenLayer,self).update_layer()
        self.Y.mean.gradient += self.grad_dict['dL_dYmean']
        self.Y.variance.gradient += self.grad_dict['dL_dYvar'][:,None]
        
    @staticmethod
    def from_TopHiddenLayer(layer, name='hiddenlayer'):
        assert isinstance(layer, TopHiddenLayer), 'The layer has to be a TopHiddenLayer!'
        
        if layer.back_constraint:          
            from deepgp.encoder.mlp import MLP
            encoder = MLP(layer.encoder.nUnits)
            encoder.param_array[:] = layer.encoder.param_array
        else:
            encoder = None
            
        return HiddenLayer(layer.layer_lower, layer.dim_up,
                           X=layer.X.mean.values, X_variance=layer.X.variance.values,
                           Z=layer.Z.values, 
                            num_inducing=layer.Z.shape[1],
                           kernel=layer.kern.copy(),
                           inference_method=None, encoder = encoder, 
                            noise_var=layer.likelihood.variance.values,
                           mpi_comm=layer.mpi_comm, mpi_root=layer.mpi_root,
                           auto_update=layer.auto_update, name=name)
        
class TopHiddenLayer(Layer):

    def __init__(self, layer_lower, dim_up, X=None, X_variance=None, Z=None,
                 num_inducing=10, kernel=None, inference_method=None,
                 noise_var=1e-2, init='rand', uncertain_inputs=True,
                 mpi_comm=None, mpi_root=0,
                 encoder=None,
                 back_constraint=True,
                 auto_update=True, name='tophiddenlayer'):
        
        self.dim_up, self.dim_down = dim_up, layer_lower.X.shape[1]
        likelihood = likelihoods.Gaussian(variance=noise_var)
        self.variationalterm = NormalPrior()
        super(TopHiddenLayer, self).__init__(layer_lower, self.dim_down,
                                             dim_up, likelihood, init=init,
                                             X=X, X_variance=X_variance, Z=Z, 
                                             num_inducing=num_inducing, kernel=kernel,
                                             inference_method=inference_method, uncertain_inputs=uncertain_inputs,
                                             mpi_comm=mpi_comm, mpi_root=mpi_root,
                                             back_constraint=back_constraint,
                                             encoder=encoder, auto_update=auto_update,
                                             name=name)

    def update_layer(self):
        super(TopHiddenLayer,self).update_layer()
        self.Y.mean.gradient += self.grad_dict['dL_dYmean']
        self.Y.variance.gradient += self.grad_dict['dL_dYvar'][:,None]
