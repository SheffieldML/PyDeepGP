
import numpy as np

from .layers import SparseGP_MPI
from GPy.core import Parameterized
from GPy.core import SparseGP
from GPy import likelihoods
from GPy import kern
from GPy.core.parameterization.variational import NormalPosterior, VariationalPosterior
from GPy.util.initialization import initialize_latent

from deepgp.util.variational import NormalEntropy,NormalPrior
from deepgp.util.parallel import reduceArrays

class MRDView(SparseGP_MPI):
    
    def __init__(self, Y, dim_down, dim_up, likelihood, MLP_dims=None, X=None, X_variance=None, init='rand',  Z=None, num_inducing=10,  kernel=None, inference_method=None, uncertain_inputs=True,mpi_comm=None, mpi_root=0, back_constraint=True, name='mrd-view'):

        self.uncertain_inputs = uncertain_inputs
        self.layer_lower = None
        self.scale = 1.

        if back_constraint:
            from .mlp import MLP
            from copy import deepcopy
            self.encoder = MLP([dim_down, int((dim_down+dim_up)*2./3.), int((dim_down+dim_up)/3.), dim_up] if MLP_dims is None else [dim_down]+deepcopy(MLP_dims)+[dim_up])
            X = self.encoder.predict(Y.mean.values if isinstance(Y, VariationalPosterior) else Y)
            X_variance = 0.0001*np.ones(X.shape)
            self.back_constraint = True
        else:
            self.back_constraint = False

        if Z is None:
            Z = np.random.rand(num_inducing, dim_up)*2-1. #np.random.permutation(X.copy())[:num_inducing]
        assert Z.shape[1] == X.shape[1]
        
        if likelihood is None: likelihood = likelihoods.Gaussian(variance=Y.var()*0.01)
        
        if uncertain_inputs: X = NormalPosterior(X, X_variance)
        if kernel is None: kernel = kern.RBF(dim_up, ARD = True)
        
        # The command below will also give the field self.X to the view.
        super(MRDView, self).__init__(X, Y, Z, kernel, likelihood, inference_method=inference_method, mpi_comm=mpi_comm, mpi_root=mpi_root, name=name)
        if back_constraint: self.link_parameter(self.encoder)

        if self.uncertain_inputs and self.back_constraint:
            from GPy import Param
            from GPy.core.parameterization.transformations import Logexp
            self.X_var_common = Param('X_var',X_variance[0].copy(),Logexp())
            self.link_parameters(self.X_var_common)
        # There's some redundancy in the self.Xv and self.X. Currently we use self.X for the likelihood part and all calculations part,
        # self.Xv is only used for the self.Xv.gradient part. 
        # This is redundant but it's there in case we want to do the product of experts MRD model.
        self.Xv = self.X


    def parameters_changed(self):
        if self.uncertain_inputs and self.back_constraint:
            self.Xv.variance[:] = self.X_var_common.values
            self.Y_encoder = (self.Y.mean.values if isinstance(self.Y, VariationalPosterior) else self.Y)
            self.Xv.mean[:] = self.encoder.predict(self.Y_encoder)
            
    def update_encoder_gradients(self):
        if self.uncertain_inputs:
            dL_dX = self.Xv.mean.gradient
            self.X_var_common.gradient = self.Xv.variance.gradient.sum(axis=0)
            Y_grad = self.encoder.update_gradient(self.Y_encoder, dL_dX)
            if self.layer_lower is not None:
                raise NotImplementedError('MRD layer has to be the observed layer at the moment!')
                self.Y.mean.gradient += Y_grad
    
    def update_qX_gradients(self):
        # Self.Xv is a placeholder for the gradient part... but we actually use the self.X for the value part (see the 'variational_posterior=' argument below)
        self.Xv.mean.gradient, self.Xv.variance.gradient = self.kern.gradients_qX_expectations(
                                            variational_posterior=self.X,
                                            Z=self.Z,
                                            dL_dpsi0=self.grad_dict['dL_dpsi0'],
                                            dL_dpsi1=self.grad_dict['dL_dpsi1'],
                                            dL_dpsi2=self.grad_dict['dL_dpsi2']) 

    def update_gradients(self):
        super(MRDView, self).parameters_changed()
        self.update_qX_gradients()
        if self.scale != 1.:
            self._log_marginal_likelihood *= self.scale
            self.gradient *= self.scale
            self.Xv.mean.gradient *= self.scale
            self.Xv.variance.gradient *= self.scale
        
    def plot_latent(self, labels=None, which_indices=None,
                resolution=50, ax=None, marker='o', s=40,
                fignum=None, plot_inducing=True, legend=True,
                plot_limits=None,
                aspect='auto', updates=False, predict_kwargs={}, imshow_kwargs={}):
        import sys
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from GPy.plotting.matplot_dep import dim_reduction_plots

        return dim_reduction_plots.plot_latent(self, labels, which_indices,
                resolution, ax, marker, s,
                fignum, plot_inducing, legend,
                plot_limits, aspect, updates, predict_kwargs, imshow_kwargs)
        

class MRDLayer(Parameterized):
    
    def __init__(self, dim_down, dim_up, likelihood, MLP_dims=None, X=None, X_variance=None, init='rand',  Z=None, num_inducing=10,  kernel=None, inference_method=None, uncertain_inputs=True,mpi_comm=None, mpi_root=0, back_constraint=True, name='mrdlayer'):

        #assert back_constraint
        self.uncertain_inputs = uncertain_inputs
        Y = self.Y if self.layer_lower is None else self.layer_lower.X
        assert isinstance(dim_down, list) or isinstance(dim_down, tuple)
        assert isinstance(kernel, list) and len(kernel)==len(dim_down), "The number of kernels has to be equal to the number of input modalities!"
        super(MRDLayer, self).__init__(name=name)
        self.mpi_comm, self.mpi_root = mpi_comm, mpi_root

        self.back_constraint = True if back_constraint else False

        self.views = []
        for i in range(len(dim_down)):
            view = MRDView(Y[i], dim_down[i],dim_up,likelihood=None if likelihood is None else likelihood[i], MLP_dims=None if MLP_dims is None else MLP_dims[i],
                           X=X, X_variance=X_variance, Z=None if Z is None else Z[i], num_inducing=num_inducing if isinstance(num_inducing,int) else num_inducing[i],
                           kernel= None if kernel is None else kernel[i], inference_method=None if inference_method is None else inference_method[i], uncertain_inputs=uncertain_inputs,
                           mpi_comm=mpi_comm, mpi_root=mpi_root, back_constraint=back_constraint, name='view_'+str(i))
            self.views.append(view)

        if self.back_constraint:
            self.X = None
            self._aggregate_qX()
        else:
            self.X = NormalPosterior(X,X_variance)

        self.link_parameters(*self.views)
        for v in self.views: v.X = self.X
        self.link_parameters(self.X)


    def _aggregate_qX(self):
        if self.back_constraint:
            if self.X is None:
                self.X = NormalPosterior(np.zeros_like(self.views[0].Xv.mean.values), np.zeros_like(self.views[0].Xv.variance.values))
            else:
                self.X.mean[:]  = 0
                self.X.variance[:] = 0
            
            self.prec_denom = np.zeros_like(self.X.variance.values)
            for v in self.views:
                self.prec_denom += 1./v.Xv.variance.values
                self.X.mean += v.Xv.mean.values/v.Xv.variance.values        
            self.X.mean /= self.prec_denom
            self.X.variance[:]  = 1./self.prec_denom
        else:
            for v in self.views:
                v.X = self.X
    
    # Wrapper function which returns single points of the observation space of this layer.
    def Y_vals(self):
        # Perhaps we shouldn't make this a function
        return self.Y if self.layer_lower is None else self.layer_lower.X.mean   

    def plot_latent(self, labels=None, which_indices=None,
                resolution=50, ax=None, marker='o', s=40,
                fignum=None, plot_inducing=True, legend=True,
                plot_limits=None,
                aspect='auto', updates=False, predict_kwargs={}, imshow_kwargs={}):
        import sys
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from GPy.plotting.matplot_dep import dim_reduction_plots

        return dim_reduction_plots.plot_latent(self, labels, which_indices,
                resolution, ax, marker, s,
                fignum, plot_inducing, legend,
                plot_limits, aspect, updates, predict_kwargs, imshow_kwargs)

    def update_qX_gradients(self):
        delta = -self.variationalterm.comp_value(self.X)
        if self.mpi_comm is not None:
            delta = reduceArrays([np.float64(delta)],self.mpi_comm, self.mpi_root)[0]
            if self.mpi_comm.rank != self.mpi_root: delta = 0
        self._log_marginal_likelihood += delta
        self.variationalterm.update_gradients(self.X)
        
    def update_encoder_gradients(self):
        if self.uncertain_inputs:
            for v in self.views:
                c = 1./(self.prec_denom*v.Xv.variance.values)
                v.Xv.mean.gradient[:] = c*self.X.mean.gradient
                v.Xv.variance.gradient[:] = np.square(c)*self.X.variance.gradient +\
                                                     c*(self.X.mean.values-v.Xv.mean.values)/v.Xv.variance.values*self.X.mean.gradient
                v.update_encoder_gradients()

    def parameters_changed(self):
        self._aggregate_qX()
        self.X.mean.gradient[:] = 0
        self.X.variance.gradient[:] = 0
        for v in self.views:
            v.update_gradients()
            self.X.mean.gradient += v.Xv.mean.gradient
            self.X.variance.gradient += v.Xv.variance.gradient

        self._log_marginal_likelihood = np.sum([v._log_marginal_likelihood for v in self.views])
        self.update_qX_gradients()

class ObservedMRDLayer(MRDLayer):

    def __init__(self, dim_down, dim_up, Y, X=None, X_variance=None, Z=None, num_inducing=10, kernel=None, inference_method=None, likelihood=None, init='rand', mpi_comm=None, mpi_root=0, MLP_dims=None, name='obslayer',back_constraint=False):
        self.layer_lower = None
        self.dim_up, self.dim_down = dim_up, dim_down
        self.Y = Y
        self._toplayer_ = False
        self.variationalterm = NormalEntropy()
        
        if not back_constraint:
            if X is None:
                # Can change self to super if we want init_X to be even for non-observed MRD layers
                X, fracs = self._init_X(Y, dim_up, init)

            if X_variance is None:
                X_variance = np.random.uniform(0,.1,X.shape)


        super(ObservedMRDLayer, self).__init__(dim_down, dim_up, likelihood, init=init, X=X, X_variance=X_variance, Z=Z, MLP_dims=MLP_dims, 
                                          num_inducing=num_inducing, kernel=kernel, inference_method=inference_method, mpi_comm=mpi_comm, mpi_root=mpi_root,  name=name, back_constraint=back_constraint)
    

    def _init_X(self, Ylist, input_dim, init='PCA'):
        if Ylist is None:
            Ylist = self.Ylist
        if init in "PCA_concat":
            print('# Initializing latent space with: PCA_concat')
            X, fracs = initialize_latent('PCA', input_dim, np.hstack(Ylist))
            fracs = [fracs]*len(Ylist)
        elif init in "PCA_single":
            print('# Initializing latent space with: PCA_single')
            X = np.zeros((Ylist[0].shape[0], input_dim))
            fracs = []
            for qs, Y in zip(np.array_split(np.arange(input_dim), len(Ylist)), Ylist):
                x,frcs = initialize_latent('PCA', len(qs), Y)
                X[:, qs] = x
                fracs.append(frcs)
        else: # init == 'random':
            print('# Initializing latent space with: random')
            X = np.random.randn(Ylist[0].shape[0], input_dim)
            fracs = X.var(0)
            fracs = [fracs]*len(Ylist)
        X -= X.mean()
        X /= X.std()
        return X, fracs        

        
    def set_as_toplayer(self, flag=True):
        if flag:
            self.variationalterm = NormalPrior()
        else:
            self.variationalterm = NormalEntropy()
        self._toplayer_ = flag

